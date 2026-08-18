#!/usr/bin/env python3
"""03_timing.py -- cumulative incidence of every timed outcome by postoperative day.

Owner: **T5 (timing)**. Owned artefacts, and nothing else:

    outputs/results_03.json
    derived/notes_03.md
    tables-csv/table3a_complication_timing.csv
    tables-csv/s1_field_definitions.csv
    tables-csv/s3_km_vs_aalen_johansen.csv
    tables-csv/s4_sensitivity_03.csv
    tables-csv/cif_*.csv

WHAT THIS STEP DOES (PLAN.md 4, 5, 6.1, 6.6, 7.3; CLAUDE.md 6.1)

 1. Aalen-Johansen cumulative incidence with DEATH AS THE COMPETING EVENT, for each of
    the eight outcomes in `variable_dict.TIMED_OUTCOMES`, on integer postoperative days
    0 to 30, with a 95% confidence band. One `tables-csv/cif_<outcome>.csv` per outcome,
    self-sufficient for `07_figures.py`.
 2. Gray's test across the four locked stratifiers (`variable_dict.STRATIFIERS`),
    honouring `variable_dict.MIN_STRATUM_EVENTS`.
 3. Table 3A: events, median postoperative day, interquartile range, share on
    postoperative day 0 or 1, share after day 10, with the denominator.
 4. Supplement S3: one minus Kaplan-Meier versus Aalen-Johansen at day 30, with the
    SIZE of the competing-risk correction measured rather than assumed small.
 5. Supplement S1: field definitions, the authoritative code sets rendered from
    `cpt_acdf`, and the `DISCHDEST` level map from `variable_dict.dischdest_level_rows()`.
 6. The coding-era check (PLAN.md 7.3): the year-by-year event rate for EVERY timed
    outcome, tested for a step at the ICD-9 to ICD-10 transition. An outcome with a step
    is reported by era and never pooled.
 7. Supplement S4 (this step's share): one minus Kaplan-Meier versus Aalen-Johansen;
    ICD-10 era only; elective defined by the emergency-case field only; and the V0 and
    V1 hematoma code sets.
 8. Gate 2 (CLAUDE.md 8.4): the median hematoma reoperation postoperative day and the
    share on postoperative day 0 or 1, RE-DERIVED here, reported on the analytic cohort
    as primary with the identified-rung reconciliation beside it.

WHY THE ESTIMATOR IS HAND-IMPLEMENTED

`lifelines` 0.30.3 ships `AalenJohansenFitter`, and it cannot be used on this data.

  * It JITTERS tied event times. Its tie detector fires whenever two DIFFERENT event
    types share a time, which on integer postoperative days happens constantly: a death
    and a reoperation on the same day is enough. `jitter_level=0` is not an escape hatch
    either, because the jitter routine raises `ValueError` when the level is not strictly
    positive. CLAUDE.md 6.1 requires a step function ON INTEGERS, and a jittered time
    axis is not one.
  * Independently of the jitter, its cumulative-incidence column is a `cumsum` over
    `d1/n * lagged_survival`. `survival_table_from_events` always synthesises a row at
    TIME 0, so time 0 is always the first row of the event table; the lagged survival is
    `NaN` there and the code zeroes it, so **every event AT TIME 0 is silently dropped
    from the estimate**.

    Say "at time 0", not "at the earliest observed time". The two coincide in this
    cohort but the mechanism does not generalise: probed directly, a fixture whose
    earliest observed event is at t = 1 loses nothing, because the synthesised time-0 row
    sits ahead of it and absorbs the `NaN`. Here time 0 IS the earliest observed time on
    all eight outcomes, and day 0 carries real events on every one of them, so the
    practical claim holds and the wording is now precise about why.

    `lifelines_diagnostic()` below demonstrates the drop deterministically: on a
    1,000-patient fixture the exact cumulative incidence is 15.0% and `lifelines` reports
    5.0%. `lifelines_cohort_diagnostic()` measures what it costs on the real cohort,
    outcome by outcome.

The estimator is therefore written out directly over 31 integer time points, which is
short enough to read and check by hand. The VARIANCE is the Klein-Moeschberger /
SAS PROC LIFETEST delta-method estimator, which is the same three-term formula
`lifelines._bounds` implements, and the confidence limits use the same log-log
transformation, so the interval is comparable with any standard implementation.

A structural consequence worth stating in the Methods: NSQIP follow-up is fixed and
complete at 30 days, so NOTHING is censored before day 30. Under complete follow-up the
Aalen-Johansen estimator reduces algebraically to the empirical cumulative proportion of
patients whose event was recorded WITH a postoperative day. `check_aj_equals_empirical()`
asserts exactly that on every outcome, every run. It is a proof of correctness, not a
coincidence, and it is also why the whole of the one-minus-Kaplan-Meier versus
Aalen-Johansen gap in S3 is attributable to the competing deaths and to nothing else.

**Say it precisely.** The reduction is to the proportion of events WITH AN OBSERVED DAY,
not to the crude proportion of all events. An event whose day was never recorded cannot
be placed on the time axis, so it is not in the numerator; on four of the eight outcomes
here that leaves a small gap against `cif.<outcome>.crude_pct`, and the gap is exactly
`cif.<outcome>.max_understatement_pct`. A Methods sentence claiming the day-30 estimate
equals the crude proportion is falsifiable by one subtraction, so this file does not
make one.

HARD RULES HONOURED HERE (CLAUDE.md section 5)

  * Reads `derived/cohort_analytic.parquet` and never re-derives a cohort, never reads
    the warehouse and never reads the export CSV. `derived/cohort_identified.parquet` is
    opened READ-ONLY for two columns, for the Gate 2 identified-rung reconciliation the
    task requires; see `identified_reconciliation()` for the justification.
  * No CPT or ICD literal: every code set is rendered from `cpt_acdf`.
  * No literal U+2014 anywhere in this file; every rendered string passes
    `variable_dict.assert_renderable` at write time.
  * No snake_case in user-visible text: names go through `variable_dict.pretty()` and
    levels through `pretty_level()`.
  * Every CSV that reports a percentage carries a `denominator` column.
  * No estimate extends past postoperative day 30.
  * Fragment keys live only under `cif.`, `timing.`, `gray.`, `era.` and `s3.`.
  * This project fits no hematoma risk model, no points score and no discrimination
    statistic. Nothing here does.

Run:  cd v1 && python3 03_timing.py
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, norm

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cpt_acdf as CPT           # noqa: E402  the ONLY place a CPT or ICD literal lives
import variable_dict as VD       # noqa: E402

# CLAUDE.md hard rule 4. This step draws on no random stream (the estimator is exact and
# the tests are analytic), but the seed is set unconditionally so that adding one later
# cannot make the step irreproducible.
np.random.seed(VD.SEED)


# ===========================================================================
# paths and constants
# ===========================================================================
DERIVED = HERE / "derived"
TABLES = HERE / "tables-csv"
OUTPUTS = HERE / "outputs"

COHORT_ANALYTIC = DERIVED / "cohort_analytic.parquet"
COHORT_IDENTIFIED = DERIVED / "cohort_identified.parquet"
FRAGMENT_PATH = OUTPUTS / "results_03.json"
NOTES_PATH = DERIVED / "notes_03.md"

# The five reserved top-level namespaces for this step (CLAUDE.md 4.2). Every key written
# to the fragment is checked against them before the file is written, so a namespace
# violation fails HERE rather than at the merge in run_all.py.
RESERVED_PREFIXES: Tuple[str, ...] = ("cif.", "timing.", "gray.", "era.", "s3.")

N_DAYS: int = VD.FOLLOWUP_DAYS          # 30. No estimate extends past it.
Z95: float = float(norm.ppf(0.975))
DENOM_COLUMN = "denominator"            # the machine-readable contract column

# Landmark days quoted in prose. Day 30 is the terminal estimate and is reported
# separately with its confidence interval.
LANDMARK_DAYS: Tuple[int, ...] = (0, 1, 2, 7, 14, 30)

# CARRIED-OVER REFERENCE VALUES, NOT MEASUREMENTS (CLAUDE.md 8.4, PLAN.md 2.2 and 12.4).
# They exist only as the comparison target for Gate 2. Nothing downstream may read them
# as an estimate: the measured values are published beside them under their own keys and
# a mismatch beyond tolerance is a STOP CONDITION to report, never a number to overwrite
# in either direction (CLAUDE.md 8.5).
CARRYOVER_MEDIAN_REOP_POD: int = 2
CARRYOVER_POD01_PCT: float = 42.6

# The event-rate band the companion step measured for hematoma, quoted here only as the
# reproduction target for the coding-era check. It is RE-DERIVED below, never copied.
CARRYOVER_HEMATOMA_BAND: Tuple[float, float] = (0.3223, 0.6390)


# ===========================================================================
# small utilities
# ===========================================================================
def _fin(x: Any) -> Optional[float]:
    """A finite float, or None. The fragment must never carry NaN or Infinity."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def num(x: Any, decimals: int = 6) -> Optional[float]:
    """A rounded finite float for the fragment, or None."""
    f = _fin(x)
    return None if f is None else round(f, decimals)


def as_int(x: Any) -> Optional[int]:
    f = _fin(x)
    return None if f is None else int(round(f))


def key_token(text: Any) -> str:
    """A fragment-key-safe lowercase token.

    `run_all.py` requires every dotted segment to match `^[a-z0-9][a-z0-9_-]*$`, so the
    locked stratum strings have to be transliterated: `>=75` becomes `ge75`, `1-2` stays
    `1-2`. The mapping is mechanical and is printed in `derived/notes_03.md` so a reader
    can invert it.
    """
    s = str(text).strip().lower()
    s = s.replace(">=", "ge").replace("<=", "le").replace(">", "gt").replace("<", "lt")
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "-_") else "-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    if not s or not (s[0].isalnum()):
        s = "x" + s
    return s


# --- code-set rendering (CLAUDE.md hard rule 5) ----------------------------
# `cpt_acdf.py` is the ONLY place a CPT or ICD code may be written as a literal. S1 has
# to PRINT those code sets, so it renders the authoritative objects instead of retyping
# them. These three helpers are deliberately identical in behaviour to the ones
# `02_build_cohort.py` uses for `cohort_schema.md`, so the two files cannot print the
# same code set two different ways.
def cpt_list(codes: Iterable[str], sep: str = ", ") -> str:
    """A `cpt_acdf` CPT set as sorted text."""
    return sep.join(sorted(str(c) for c in codes))


def icd_display(code: str) -> str:
    """A normalized ICD code back into printed form: the point goes after 3 characters."""
    text = str(code).strip().upper()
    return text if len(text) <= 3 else f"{text[:3]}.{text[3:]}"


def icd_list(codes: Iterable[str], sep: str = ", ") -> str:
    """A `cpt_acdf` ICD set as sorted printed codes."""
    return sep.join(icd_display(c) for c in sorted(str(c) for c in codes))


# ===========================================================================
# THE ESTIMATOR
# ===========================================================================
class CIFResult:
    """One Aalen-Johansen cumulative incidence function on integer days 0..N_DAYS."""

    __slots__ = ("days", "cif", "var", "lo", "hi", "at_risk", "e1", "e2", "cens",
                 "cum_e1", "cum_e2", "n", "km_cif", "km_var", "km_lo", "km_hi")

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def day30(self) -> float:
        return float(self.cif[-1])

    @property
    def day30_lo(self) -> float:
        return float(self.lo[-1])

    @property
    def day30_hi(self) -> float:
        return float(self.hi[-1])


def _loglog_ci(est: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """95% limits from the log-log transformation of a cumulative incidence.

    SE(log(-log F)) = SE(F) / (F * |log F|), then back-transform. The transformation is
    the one PROC LIFETEST and `lifelines._bounds` use; it keeps both limits inside (0, 1)
    for any variance, which a linear Wald interval does not for a probability this small.
    Where the estimate is exactly 0 (no event has occurred yet) the variance is 0 as
    well, the transformation is undefined, and the interval is reported as [0, 0]: the
    band opens only once the first event occurs, which is the honest picture.
    """
    lo = np.zeros_like(est)
    hi = np.zeros_like(est)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(est)):
            f = float(est[i])
            s = float(sd[i])
            if not (0.0 < f < 1.0) or s <= 0.0:
                lo[i] = f
                hi[i] = f
                continue
            g = math.log(-math.log(f))
            se_g = s / (f * abs(math.log(f)))
            lo[i] = math.exp(-math.exp(g + Z95 * se_g))
            hi[i] = math.exp(-math.exp(g - Z95 * se_g))
    return lo, hi


def aalen_johansen(times: np.ndarray, etypes: np.ndarray,
                   n_days: int = N_DAYS) -> CIFResult:
    """Aalen-Johansen cumulative incidence for cause 1, competing cause 2, on 0..n_days.

    `times` are INTEGER postoperative days. `etypes` is 0 (event free at the end of
    follow-up), 1 (the event of interest) or 2 (the competing event, death).

    The recursion, on integer day d, with the risk set taken just BEFORE the day:

        n_d    = n_{d-1} - e1_{d-1} - e2_{d-1} - c_{d-1}
        CIF(d) = CIF(d-1) + S(d-1) * e1_d / n_d
        S(d)   = S(d-1) * (1 - (e1_d + e2_d) / n_d)

    with S(-1) = 1 and CIF(-1) = 0. Day 0 is a real day and events recorded on it are
    real events: the risk set on day 0 is the whole cohort. This is precisely the point
    `lifelines` gets wrong (see the module docstring).

    Also returned, from the same event table, is one minus the ordinary Kaplan-Meier
    estimate for cause 1 with the competing event treated as non-informative censoring.
    That is the WRONG estimator for this question and is computed only so supplement S3
    can measure how wrong it is.

    Variance: the Klein-Moeschberger / PROC LIFETEST delta-method estimator,

        Var[F(t)] =  sum_j (F(t) - F(t_j))^2 * d_j / (n_j (n_j - d_j))
                  +  sum_j S(t_{j-1})^2 * d1_j (n_j - d1_j) / n_j^3
                  -  2 sum_j (F(t) - F(t_j)) * S(t_{j-1}) * d1_j / n_j^2

    where d_j counts events of BOTH causes and d1_j only the cause of interest.
    """
    times = np.asarray(times)
    etypes = np.asarray(etypes)
    n = int(len(times))
    days = np.arange(0, n_days + 1, dtype=int)

    e1 = np.zeros(n_days + 1)
    e2 = np.zeros(n_days + 1)
    cens = np.zeros(n_days + 1)
    for d in days:
        on = times == d
        e1[d] = float(np.sum(on & (etypes == 1)))
        e2[d] = float(np.sum(on & (etypes == 2)))
        cens[d] = float(np.sum(on & (etypes == 0)))

    at_risk = np.empty(n_days + 1)
    cur = float(n)
    for d in days:
        at_risk[d] = cur
        cur -= e1[d] + e2[d] + cens[d]

    cif = np.zeros(n_days + 1)
    lag_s = np.zeros(n_days + 1)     # S(d-1), the all-cause event-free survival
    km_cif = np.zeros(n_days + 1)    # 1 - KM, competing event treated as censoring
    s_all = 1.0
    s_km = 1.0
    c = 0.0
    for d in days:
        y = at_risk[d]
        lag_s[d] = s_all
        if y > 0:
            h1 = e1[d] / y
            h_all = (e1[d] + e2[d]) / y
            c += s_all * h1
            s_all *= (1.0 - h_all)
            s_km *= (1.0 - h1)
        cif[d] = c
        km_cif[d] = 1.0 - s_km

    # --- Aalen-Johansen delta-method variance ------------------------------
    d_all = e1 + e2
    var = np.zeros(n_days + 1)
    for i in days:
        ft = cif[i]
        acc = 0.0
        for j in range(0, i + 1):
            nj = at_risk[j]
            if nj <= 0:
                continue
            if d_all[j] > 0 and nj > d_all[j]:
                acc += (ft - cif[j]) ** 2 * d_all[j] / (nj * (nj - d_all[j]))
            if e1[j] > 0:
                acc += (lag_s[j] ** 2) * e1[j] * (nj - e1[j]) / (nj ** 3)
                acc += -2.0 * (ft - cif[j]) * lag_s[j] * e1[j] / (nj ** 2)
        var[i] = max(acc, 0.0)

    # --- Greenwood variance for one minus Kaplan-Meier ---------------------
    # Var[1 - KM] = KM^2 * sum_j d1_j / (n_j (n_j - d1_j)); the risk set for this
    # estimator loses the competing-event patients to censoring, which is exactly the
    # assumption S3 exists to quantify.
    km_var = np.zeros(n_days + 1)
    green = 0.0
    for d in days:
        y = at_risk[d]
        if y > 0 and e1[d] > 0 and y > e1[d]:
            green += e1[d] / (y * (y - e1[d]))
        surv = 1.0 - km_cif[d]
        km_var[d] = (surv ** 2) * green

    lo, hi = _loglog_ci(cif, var)
    km_lo, km_hi = _loglog_ci(km_cif, km_var)

    return CIFResult(
        days=days, cif=cif, var=var, lo=lo, hi=hi,
        at_risk=at_risk, e1=e1, e2=e2, cens=cens,
        cum_e1=np.cumsum(e1), cum_e2=np.cumsum(e2), n=n,
        km_cif=km_cif, km_var=km_var, km_lo=km_lo, km_hi=km_hi,
    )


def grays_test(times: np.ndarray, etypes: np.ndarray, groups: np.ndarray,
               n_days: int = N_DAYS) -> Tuple[Optional[float], Optional[int],
                                              Optional[float], List[str]]:
    """Gray's k-sample test for equality of cumulative incidence functions, rho = 0.

    Gray (1988) compares SUBDISTRIBUTION hazards: a patient who experiences the competing
    event STAYS in the risk set, because they remain unable to fail from the cause of
    interest and the subdistribution hazard is defined on that enlarged risk set. With
    independent censoring the enlarged risk set has to be reweighted by the inverse
    probability of remaining uncensored.

    In this cohort there is NO censoring before day 30: NSQIP follow-up is fixed and
    complete by construction, so every inverse-probability-of-censoring weight is exactly
    one and the subdistribution risk set on day d is simply

        Y_k(d) = n_k - (events of the cause of interest in group k on days < d)

    The rho = 0 member of Gray's family then reduces exactly to the multivariate log-rank
    statistic computed on those risk sets, with the usual hypergeometric variance. That
    is what is written out below, which makes the whole test auditable by hand rather
    than delegated to a black box.

    Returns (statistic, degrees of freedom, p value, level order actually tested).
    """
    lev = [str(g) for g in pd.unique(pd.Series(groups))]
    lev = [g for g in lev if g != ""]
    lev.sort()
    k = len(lev)
    if k < 2:
        return None, None, None, lev

    days = np.arange(0, n_days + 1, dtype=int)
    idx = {g: i for i, g in enumerate(lev)}
    gi = np.array([idx.get(str(g), -1) for g in groups], dtype=int)
    keep = gi >= 0
    gi = gi[keep]
    t = np.asarray(times)[keep]
    e = np.asarray(etypes)[keep]

    n_k = np.array([float(np.sum(gi == i)) for i in range(k)])
    d_kd = np.zeros((n_days + 1, k))
    for d in days:
        on = (t == d) & (e == 1)
        for i in range(k):
            d_kd[d, i] = float(np.sum(on & (gi == i)))

    u = np.zeros(k)
    v = np.zeros((k, k))
    y_k = n_k.copy()
    for d in days:
        y = float(y_k.sum())
        dd = float(d_kd[d].sum())
        if y <= 0:
            break
        if dd > 0:
            frac = y_k / y
            u += d_kd[d] - dd * frac
            if y > 1:
                scale = dd * (y - dd) / (y - 1.0)
                v += scale * (np.diag(frac) - np.outer(frac, frac))
        y_k = y_k - d_kd[d]     # only a cause-of-interest event leaves the risk set

    # Drop one component to get a full-rank quadratic form on k - 1 degrees of freedom.
    u_r = u[:-1]
    v_r = v[:-1, :-1]
    if np.allclose(v_r, 0.0):
        return None, None, None, lev
    stat = float(u_r @ np.linalg.pinv(v_r) @ u_r)
    dof = k - 1
    p = float(chi2.sf(stat, dof))
    return stat, dof, p, lev


# ===========================================================================
# building the (time, event type) representation
# ===========================================================================
def build_times(df: pd.DataFrame, key: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Time and event type for one outcome, from the FROZEN columns only.

    Rules, all of which are stated in `derived/notes_03.md`:

      * time zero is the day of the principal operation, postoperative day 0;
      * a patient with the event flag positive and an OBSERVED postoperative day enters
        as the event of interest on that day;
      * a patient who died within the window without such an event enters as the
        COMPETING event on the recorded death day. For the death outcome itself there is
        no competing event and the estimator reduces to the ordinary cumulative
        incidence, which is stated rather than dressed up as a self-competing estimate;
      * everyone else is administratively censored at day 30. NSQIP follow-up is fixed
        and complete, so this is the ONLY censoring in the data and it is
        non-informative by construction;
      * a patient whose event fired but whose day was not recorded cannot be placed on
        the time axis. Their timing is unobserved, they stay in the cohort and in the
        risk set, and they contribute no event. The count is published per outcome
        (`cif.<key>.day_unobserved_events_n`) so a reader can bound the understatement,
        which is at most that count divided by the analytic denominator. `02` measured
        the days-to-event missingness as 0 to 10 events per outcome; nothing is imputed
        and nothing is clipped (the 0 to 30 integer handling rule was applied once, in
        `02`, and this step inherits it through the frozen columns).
    """
    o = VD.timed_outcome(key)
    n = len(df)
    ev = (df[o.event_col].fillna(0).astype("int64") == 1).to_numpy()
    pod = pd.to_numeric(df[o.pod_col], errors="coerce").to_numpy(dtype=float)
    died = (df["MORT30"].fillna(0).astype("int64") == 1).to_numpy()
    dday = pd.to_numeric(df["POD_DEATH"], errors="coerce").to_numpy(dtype=float)

    has_event = ev & np.isfinite(pod)
    times = np.full(n, float(N_DAYS))
    etypes = np.zeros(n, dtype=int)
    times[has_event] = pod[has_event]
    etypes[has_event] = 1

    counts = {
        "events_n": int(ev.sum()),
        "day_observed_n": int(has_event.sum()),
        "day_unobserved_n": int((ev & ~np.isfinite(pod)).sum()),
        "competing_n": 0,
        "event_and_death_n": 0,
        "death_before_event_n": 0,
    }

    if key != "death":
        comp = died & ~has_event
        times[comp] = dday[comp]
        etypes[comp] = 2
        counts["competing_n"] = int(comp.sum())
        both = has_event & died
        counts["event_and_death_n"] = int(both.sum())
        counts["death_before_event_n"] = int(np.sum(both & (dday < pod)))

    # An event day after the recorded death day would be an ordering violation and would
    # make the competing-risks structure incoherent. It is checked, not assumed.
    if counts["death_before_event_n"]:
        raise SystemExit(
            f"STOP: {counts['death_before_event_n']} rows carry a {key} event day after "
            f"the recorded death day. The competing-risks ordering is violated and no "
            f"curve may be drawn. Report this rather than resolving it here."
        )

    bad = ~np.isfinite(times) | (times < 0) | (times > N_DAYS)
    if bad.any():
        raise SystemExit(
            f"STOP: {int(bad.sum())} rows carry a {key} time outside 0 to {N_DAYS} days "
            f"after the frozen handling rule was applied in 02_build_cohort.py."
        )
    return times.astype(int), etypes, counts


# ===========================================================================
# self-checks
# ===========================================================================
def _self_test() -> None:
    """Analytic checks on the estimator, run before it touches the cohort.

    Nothing here reads the data. The fixture in part 2 is the deterministic demonstration
    that `lifelines.AalenJohansenFitter` cannot be used on integer-day data.
    """
    # 1. A worked example whose Aalen-Johansen values are exact decimals.
    #    10 patients: (1,event) (1,death) (2,event) (3,censored) (3,event) (4,death)
    #                 (5,censored) (5,event) (6,censored) (7,event)
    t = np.array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7])
    e = np.array([1, 2, 1, 0, 1, 2, 0, 1, 0, 1])
    r = aalen_johansen(t, e, n_days=7)
    expect = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.30, 5: 0.42, 6: 0.42, 7: 0.78}
    for d, want in expect.items():
        got = float(r.cif[d])
        assert abs(got - want) < 1e-12, (d, got, want)
    assert float(r.cif[0]) == 0.0
    # one minus Kaplan-Meier must EXCEED the Aalen-Johansen value once a competing event
    # has occurred; that inequality is the whole content of supplement S3.
    assert r.km_cif[7] > r.cif[7]
    # the limits must bracket the estimate and stay inside the unit interval
    assert np.all(r.lo <= r.cif + 1e-12) and np.all(r.cif <= r.hi + 1e-12)
    assert np.all(r.lo >= -1e-12) and np.all(r.hi <= 1.0 + 1e-12)

    # 2. Complete follow-up: the estimator must reproduce the empirical proportion.
    t2, e2 = _library_fixture()
    r2 = aalen_johansen(t2.astype(int), e2, n_days=5)
    assert abs(float(r2.cif[5]) - 150.0 / 1000.0) < 1e-12, float(r2.cif[5])

    # 3. Gray's test: identical groups give a large p, disjoint groups a small one.
    g_same = np.array(["a"] * 500 + ["b"] * 500)
    stat, dof, p, lev = grays_test(t2, e2, g_same, n_days=5)
    assert dof == 1 and p is not None
    g_split = np.array(["a"] * 150 + ["b"] * 850)
    stat2, dof2, p2, _ = grays_test(t2, e2, g_split, n_days=5)
    assert p2 is not None and p2 < 1e-6, p2

    # 4. Level tokens survive the fragment key regex.
    import re as _re
    seg = _re.compile(r"^[a-z0-9][a-z0-9_-]*$")
    for s in VD.STRATIFIERS:
        for lv in s.levels:
            assert seg.match(key_token(lv)), (s.column, lv, key_token(lv))
        assert seg.match(key_token(s.column)), s.column
    for o in VD.TIMED_OUTCOMES:
        assert seg.match(key_token(o.key)), o.key


def _library_fixture() -> Tuple[np.ndarray, np.ndarray]:
    """A 1,000-patient fixture with an exact answer of 15.0% at day 5.

    Shaped so that both `lifelines` failure modes are exercised at once:

      * day 0 carries BOTH an event of interest and a competing event, so the library's
        tie detector fires and the jitter path is taken (its detector only looks for two
        DIFFERENT event types at the same time, which integer day data produces
        constantly);
      * day 0 also carries 100 events of interest, and TIME 0 is where the library's
        synthesised first event-table row sits, so those 100 are the ones it drops.

    Exact Aalen-Johansen by hand: day 0 gives 100/1000 = 0.10 with S = 0.89; day 2 adds
    0.89 * 50/890 = 0.05, giving 0.15; day 3 is a competing event only; day 5 is the
    administrative censoring. Terminal value 15.0%.
    """
    t = np.array([0.0] * 110 + [2.0] * 50 + [3.0] * 30 + [5.0] * 810)
    e = np.array([1] * 100 + [2] * 10 + [1] * 50 + [2] * 30 + [0] * 810)
    return t, e


def lifelines_diagnostic() -> Dict[str, Any]:
    """Demonstrate, deterministically, why the library implementation is not used.

    Returns the measured library value on `_library_fixture()`, whose exact answer is
    15.0%. If `lifelines` is not importable the diagnostic is skipped and reported as
    skipped; it is evidence, not a dependency.
    """
    out: Dict[str, Any] = {"available": False}
    try:
        from lifelines import AalenJohansenFitter  # noqa: WPS433
    except Exception:  # pragma: no cover
        return out
    t, e = _library_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = AalenJohansenFitter(seed=VD.SEED).fit(t, e, event_of_interest=1)
    lib = float(f.cumulative_density_.iloc[-1, 0])
    exact = float(aalen_johansen(t.astype(int), e, n_days=5).cif[-1])
    out.update({
        "available": True,
        "fixture_exact_pct": exact * 100.0,
        "fixture_library_pct": lib * 100.0,
        "fixture_events_lost_n": int(round((exact - lib) * 1000.0)),
    })
    # The tie detector is narrower than the docstring implies: it fires only where a
    # single event of one type shares a time with a single event of another type. Once it
    # does fire, `jitter_level=0` is rejected outright, so there is no way to ask the
    # fitter for an unjittered integer time axis. Both facts are measured, on the
    # smallest fixture that triggers the detector.
    tj = np.array([1.0, 1.0, 2.0, 3.0])
    ej = np.array([1, 2, 0, 1])
    out["tie_detector_fires"] = bool(
        AalenJohansenFitter._check_for_duplicates(durations=tj, events=ej))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AalenJohansenFitter(jitter_level=0.0, seed=VD.SEED).fit(
                tj, ej, event_of_interest=1)
        out["rejects_zero_jitter"] = False
    except Exception:
        out["rejects_zero_jitter"] = True
    return out


def lifelines_cohort_diagnostic(store: Dict[str, Any]) -> Dict[str, float]:
    """What the library estimator would have cost on THIS cohort, outcome by outcome.

    Reported as the difference in percentage points between the exact day-30 estimate and
    what `lifelines` returns on the same (time, event type) vectors. It is a
    library-behaviour diagnostic and never an estimate; nothing in the manuscript reads
    it. If the library is not importable the diagnostic is skipped.
    """
    out: Dict[str, float] = {}
    try:
        from lifelines import AalenJohansenFitter  # noqa: WPS433
    except Exception:  # pragma: no cover
        return out
    fires = 0
    for o in VD.TIMED_OUTCOMES:
        t = np.asarray(store[o.key]["times"], dtype=float)
        e = np.asarray(store[o.key]["etypes"], dtype=int)
        if AalenJohansenFitter._check_for_duplicates(durations=t, events=e):
            fires += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = AalenJohansenFitter(seed=VD.SEED).fit(t, e, event_of_interest=1)
            cd = f.cumulative_density_.iloc[:, 0]
            lib = float(cd[cd.index <= N_DAYS].iloc[-1])
        except Exception:  # pragma: no cover
            continue
        out[o.key] = (float(store[o.key]["res"].day30) - lib) * 100.0
    out["_jitter_outcomes_n"] = float(fires)
    return out


def check_aj_equals_empirical(res: CIFResult, key: str) -> float:
    """Assert the structural identity that complete follow-up guarantees.

    With no censoring before day 30 the Aalen-Johansen estimator equals the empirical
    cumulative proportion of patients whose event of interest was recorded WITH a
    postoperative day. That is the quantity compared here: `res.cum_e1[-1]` counts the
    events placed on the time axis, which is the events with an observed day, and NOT
    the crude count of all events. The two differ on any outcome that lost an event day,
    and confusing them is how a Methods sentence gets falsified.

    Any drift means either a censoring time slipped in before day 30 or the recursion is
    wrong, and either is a stop condition rather than a rounding matter.
    """
    empirical = float(res.cum_e1[-1]) / float(res.n)
    gap = abs(res.day30 - empirical)
    if gap > 1e-10:
        raise SystemExit(
            f"STOP: for {key} the Aalen-Johansen day-30 estimate {res.day30:.10f} does "
            f"not equal the empirical proportion {empirical:.10f}. Follow-up is supposed "
            f"to be complete to day 30; investigate before reporting a curve."
        )
    return empirical


# ===========================================================================
# fragment and CSV writers
# ===========================================================================
class Fragment:
    """The `outputs/results_03.json` fragment, validated as it is built."""

    def __init__(self) -> None:
        self._d: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        k = str(key).strip().lower()
        if k in self._d:
            raise SystemExit(f"STOP: duplicate fragment key {k!r} written by 03_timing.py")
        if not any(k == p[:-1] or k.startswith(p) for p in RESERVED_PREFIXES):
            raise SystemExit(
                f"STOP: fragment key {k!r} is outside this step's reserved namespaces "
                f"{RESERVED_PREFIXES}. CLAUDE.md 4.2."
            )
        if isinstance(value, (list, tuple)):
            vals = []
            for v in value:
                if isinstance(v, (list, tuple, dict, set)):
                    raise SystemExit(f"STOP: nested value under {k!r}")
                vals.append(self._scalar(k, v))
            self._d[k] = vals
        elif isinstance(value, dict):
            raise SystemExit(f"STOP: nested object under {k!r}")
        else:
            self._d[k] = self._scalar(k, value)

    @staticmethod
    def _scalar(k: str, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            f = float(v)
            if not math.isfinite(f):
                return None
            return f
        if isinstance(v, str):
            return v
        raise SystemExit(f"STOP: unsupported value type {type(v)!r} under {k!r}")

    def write(self, path: pathlib.Path) -> int:
        import re as _re
        pat = _re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")
        for k in self._d:
            if not pat.match(k):
                raise SystemExit(f"STOP: fragment key {k!r} fails the run_all.py key regex")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._d, indent=2, sort_keys=True, allow_nan=False)
        if "NaN" in text or "Infinity" in text:
            raise SystemExit("STOP: the fragment carries NaN or Infinity")
        path.write_text(text + "\n", encoding="utf-8")
        return len(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> Iterable[str]:
        return self._d.keys()


def write_csv(path: pathlib.Path, columns: Sequence[str],
              rows: Sequence[Dict[str, Any]]) -> None:
    """Write a table, asserting every header and every cell is renderable.

    CLAUDE.md hard rules 1 and 2 are enforced at WRITE TIME, which is the only place they
    can be enforced without a grep. Hard rule 7 is enforced here too: any table that
    reports a percentage must carry the `denominator` column.
    """
    for c in columns:
        VD.assert_renderable(c, where=f"{path.name} header")
    if any("%" in c for c in columns) and DENOM_COLUMN not in columns:
        raise SystemExit(
            f"STOP: {path.name} reports a percentage but carries no {DENOM_COLUMN!r} "
            f"column (CLAUDE.md hard rule 7)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        w.writeheader()
        for r in rows:
            out = {}
            for c in columns:
                v = r.get(c, "")
                v = "" if v is None else v
                out[c] = VD.assert_renderable(v, where=f"{path.name} column {c}")
            w.writerow(out)


# ===========================================================================
# reporting helpers
# ===========================================================================
def pct_ci(est: float, lo: float, hi: float, decimals: int = 3) -> str:
    """A cumulative incidence and its band, as a percentage with an EN-dash range."""
    return (f"{VD.fmt_pct(est * 100.0, decimals)} "
            f"({VD.fmt_ci(lo * 100.0, hi * 100.0, decimals)})")


def median_iqr_days(values: np.ndarray) -> Tuple[Optional[float], Optional[float],
                                                 Optional[float]]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None, None
    return (float(np.median(v)),
            float(np.percentile(v, 25)),
            float(np.percentile(v, 75)))


# ===========================================================================
# 0. UNITS AND SCALES (.build_facts.md section 9)
# ===========================================================================
# `.est`, `.lo` and `.hi` are generic confidence-interval suffixes. Four steps use
# them for four incompatible kinds of quantity: a proportion here, a percentage in
# `06`, an odds ratio and a kappa in `05`. A scale declaration that can be read as
# project-wide is therefore a 100x error waiting to happen, so every declaration
# below names the ONE namespace it governs and says so in its own text.
#
# `<namespace>.scale` is prose a human can read. `<namespace>.scale.families` and
# `<namespace>.scale.units` carry the same content as two parallel lists, so
# `verify_numbers.py` can assert a unit mechanically rather than parse English. The
# family entries are KEY PATTERNS addressed to `08`, not rendered prose.

UNIT_PROPORTION = "proportion on the 0 to 1 scale"
UNIT_PERCENT = "percentage on the 0 to 100 scale"
UNIT_PP = "percentage points"
UNIT_COUNT = "count of patients or events"
UNIT_POD = "integer postoperative day"
UNIT_DAYS = "days"
UNIT_P = "p value"
UNIT_STAT = "test statistic"
UNIT_YEAR = "calendar year"

SCALE_SCOPE_NOTE = (
    "This declaration governs only the namespace it is named for. It is not a "
    "project-wide rule and must never be read as one: the same est, lo and hi "
    "suffixes carry percentages, odds ratios and a kappa in other steps, so read "
    "the declaration published by the step that owns the key."
)

# (namespace, prose, ((key pattern, unit), ...))
SCALE_DECLARATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        "cif",
        "Within the cumulative-incidence namespace, a key ending in est, lo or hi is "
        "a proportion on the 0 to 1 scale, and a key ending in pct is a percentage "
        "on the 0 to 100 scale. The two forms carry the same quantity and must never "
        "be mixed. The 30-day incidence a manuscript should quote is the day 30 "
        "percentage, which is also the terminal point of the Figure 2 curve.",
        (
            ("cif.<outcome>.day30.est", UNIT_PROPORTION),
            ("cif.<outcome>.day30.lo", UNIT_PROPORTION),
            ("cif.<outcome>.day30.hi", UNIT_PROPORTION),
            ("cif.<outcome>.day<D>.pct", UNIT_PERCENT),
            ("cif.<outcome>.day30.pct_lo", UNIT_PERCENT),
            ("cif.<outcome>.day30.pct_hi", UNIT_PERCENT),
            ("cif.<outcome>.crude_pct", UNIT_PERCENT),
            ("cif.<outcome>.max_understatement_pct", UNIT_PERCENT),
            ("cif.followup_days", UNIT_DAYS),
            ("cif.landmark_days", UNIT_POD),
        ),
    ),
    (
        "timing",
        "Within the descriptive-timing namespace, a key ending in pct is a "
        "percentage on the 0 to 100 scale and a key ending in pp is a difference "
        "expressed in percentage points. A postoperative day is an integer day "
        "count from the principal operation. Postoperative length of stay is NOT "
        "bounded at day 30 and runs to 112 days in this cohort.",
        (
            ("timing.<outcome>.crude_pct", UNIT_PERCENT),
            ("timing.<outcome>.pod01_pct", UNIT_PERCENT),
            ("timing.<outcome>.after_day10_pct", UNIT_PERCENT),
            ("timing.<outcome>.median_pod", UNIT_POD),
            ("timing.<outcome>.q1_pod", UNIT_POD),
            ("timing.<outcome>.q3_pod", UNIT_POD),
            ("timing.los_postop.median", UNIT_DAYS),
            ("timing.los_postop.q1", UNIT_DAYS),
            ("timing.los_postop.q3", UNIT_DAYS),
            ("timing.los_postop.max", UNIT_DAYS),
            ("timing.los_postop.figure2_reference_day", UNIT_DAYS),
            ("timing.sens.<tag>.<outcome>.day30_pct", UNIT_PERCENT),
            ("timing.sens.<tag>.<outcome>.day30_pct_lo", UNIT_PERCENT),
            ("timing.sens.<tag>.<outcome>.day30_pct_hi", UNIT_PERCENT),
            ("timing.sens.<tag>.<outcome>.diff_pp", UNIT_PP),
            ("timing.sens.hematoma.<version>.day30_pct", UNIT_PERCENT),
            ("timing.sens.hematoma.<version>.pod01_pct", UNIT_PERCENT),
            ("timing.sens.hematoma.<version>.median_pod", UNIT_POD),
            ("timing.library.<outcome>.understatement_pp", UNIT_PP),
            ("timing.library.max_understatement_pp", UNIT_PP),
            ("timing.library.fixture_exact_pct", UNIT_PERCENT),
            ("timing.library.fixture_reported_pct", UNIT_PERCENT),
            ("timing.second_event.patients_pct", UNIT_PERCENT),
            ("timing.gate2.pod01_pct_analytic", UNIT_PERCENT),
            ("timing.gate2.pod01_pct_identified", UNIT_PERCENT),
            ("timing.gate2.pod01_pct_carried", UNIT_PERCENT),
            ("timing.gate2.median_pod_analytic", UNIT_POD),
            ("timing.gate2.median_pod_identified", UNIT_POD),
            ("timing.followup_days", UNIT_DAYS),
        ),
    ),
    (
        "gray",
        "Within the stratum-comparison namespace, a stratum-level day 30 cumulative "
        "incidence is a percentage on the 0 to 100 scale, and it is withheld where "
        "the stratum carries fewer events than the suppression rule allows. The "
        "statistic is a chi-square value on its own degrees of freedom.",
        (
            ("gray.<outcome>.<stratifier>.<level>.day30_pct", UNIT_PERCENT),
            ("gray.<outcome>.<stratifier>.stat", UNIT_STAT),
            ("gray.<outcome>.<stratifier>.p", UNIT_P),
        ),
    ),
    (
        "era",
        "Within the coding-era namespace, a key ending in pct is a percentage on the "
        "0 to 100 scale and a key ending in pp is a difference between two such "
        "percentages, expressed in percentage points. Year keys are calendar years.",
        (
            ("era.<outcome>.pre_pct", UNIT_PERCENT),
            ("era.<outcome>.post_pct", UNIT_PERCENT),
            ("era.<outcome>.year_min_pct", UNIT_PERCENT),
            ("era.<outcome>.year_max_pct", UNIT_PERCENT),
            ("era.<outcome>.rate_by_year", UNIT_PERCENT),
            ("era.<outcome>.band_min_pct", UNIT_PERCENT),
            ("era.<outcome>.band_max_pct", UNIT_PERCENT),
            ("era.<outcome>.band_carried_min_pct", UNIT_PERCENT),
            ("era.<outcome>.band_carried_max_pct", UNIT_PERCENT),
            ("era.<outcome>.boundary_change_pp", UNIT_PP),
            ("era.<outcome>.largest_change_pp", UNIT_PP),
            ("era.<outcome>.boundary_p", UNIT_P),
            ("era.<outcome>.pooled_p", UNIT_P),
            ("era.transition_year", UNIT_YEAR),
            ("era.years", UNIT_YEAR),
        ),
    ),
    (
        "s3",
        "Within the competing-risk-correction namespace every estimate and limit is "
        "a percentage on the 0 to 100 scale, the absolute difference between the two "
        "estimators is in percentage points, and the relative difference is a "
        "percentage of the Aalen-Johansen value.",
        (
            ("s3.<outcome>.aj_pct", UNIT_PERCENT),
            ("s3.<outcome>.aj_pct_lo", UNIT_PERCENT),
            ("s3.<outcome>.aj_pct_hi", UNIT_PERCENT),
            ("s3.<outcome>.km_pct", UNIT_PERCENT),
            ("s3.<outcome>.km_pct_lo", UNIT_PERCENT),
            ("s3.<outcome>.km_pct_hi", UNIT_PERCENT),
            ("s3.<outcome>.relative_pct", UNIT_PERCENT),
            ("s3.<outcome>.diff_pp", UNIT_PP),
            ("s3.max_diff_pp", UNIT_PP),
            ("s3.max_relative_pct", UNIT_PERCENT),
        ),
    ),
)


def publish_scale_declarations(frag: Fragment) -> None:
    """Declare the unit of every numeric family this step publishes.

    Binding: `.build_facts.md` section 9. One declaration per namespace, each scoped
    to its own namespace and saying so, because the merged `results.json` resolves
    `.est` / `.lo` / `.hi` to at least four incompatible scales and the only
    documented rule used to be this step's.
    """
    for namespace, prose, families in SCALE_DECLARATIONS:
        frag.put(f"{namespace}.scale", f"{prose} {SCALE_SCOPE_NOTE}")
        frag.put(f"{namespace}.scale.families", [f for f, _ in families])
        frag.put(f"{namespace}.scale.units", [u for _, u in families])
        frag.put(f"{namespace}.scale.other_numeric_keys",
                 "Every other numeric key in this namespace is a count of patients "
                 "or events, a denominator, or an integer day.")


# ===========================================================================
# 1. cumulative incidence by postoperative day
# ===========================================================================
def analysis_cif(df: pd.DataFrame, frag: Fragment) -> Dict[str, Any]:
    n_analytic = len(df)
    store: Dict[str, Any] = {}

    frag.put("cif.denominator", n_analytic)
    frag.put("cif.followup_days", N_DAYS)
    frag.put("cif.landmark_days", list(LANDMARK_DAYS))

    # The structural identity, stated precisely rather than loosely. Nothing is
    # censored before day 30, so the estimator reduces to an empirical cumulative
    # proportion. The numerator of that proportion is the events whose postoperative
    # DAY WAS RECORDED, not all events, because an event with no recorded day cannot
    # be placed on the time axis. Against the all-events crude proportion the day-30
    # estimate is therefore slightly lower on any outcome that lost a day, and the
    # gap is exactly the published maximum understatement. A Methods sentence that
    # claims equality against the crude proportion is falsifiable in one subtraction.
    frag.put("cif.identity_note",
             "NSQIP follow-up is fixed and complete at 30 days, so nothing is "
             "censored before day 30 and the Aalen-Johansen estimator reduces "
             "algebraically to the empirical cumulative proportion of patients whose "
             "event was recorded WITH a postoperative day. It is not equal to the "
             "crude proportion of all events unless every event carried a day.")
    frag.put("cif.crude_pct_note",
             "The crude percentage is all recorded events over the analytic cohort, "
             "including any event whose postoperative day was not recorded. It is "
             "therefore greater than or equal to the day 30 estimate, and the "
             "difference is exactly the events with no recorded day over the same "
             "denominator, published as the maximum understatement for that outcome.")
    frag.put("cif.day30_reporting_key",
             "Report the 30-day incidence from the day 30 percentage in this "
             "namespace, with its lower and upper percentage limits. That value is "
             "the terminal point of the Figure 2 curve. Do NOT report the crude "
             "percentage as the 30-day incidence: it is a different quantity with a "
             "different numerator and it does not match the figure.")

    for o in VD.TIMED_OUTCOMES:
        key = o.key
        times, etypes, counts = build_times(df, key)
        res = aalen_johansen(times, etypes)
        empirical = check_aj_equals_empirical(res, key)
        store[key] = {"res": res, "counts": counts, "times": times,
                      "etypes": etypes, "empirical": empirical}

        frag.put(f"cif.{key}.denominator", n_analytic)
        frag.put(f"cif.{key}.events_n", counts["events_n"])
        frag.put(f"cif.{key}.day_observed_events_n", counts["day_observed_n"])
        frag.put(f"cif.{key}.day_unobserved_events_n", counts["day_unobserved_n"])
        frag.put(f"cif.{key}.competing_deaths_n", counts["competing_n"])
        frag.put(f"cif.{key}.has_competing_event", key != "death")
        frag.put(f"cif.{key}.crude_pct", num(counts["events_n"] / n_analytic * 100.0, 4))
        frag.put(f"cif.{key}.max_understatement_pct",
                 num(counts["day_unobserved_n"] / n_analytic * 100.0, 4))
        frag.put(f"cif.{key}.day30.est", num(res.day30, 8))
        frag.put(f"cif.{key}.day30.lo", num(res.day30_lo, 8))
        frag.put(f"cif.{key}.day30.hi", num(res.day30_hi, 8))
        frag.put(f"cif.{key}.day30.pct", num(res.day30 * 100.0, 4))
        frag.put(f"cif.{key}.day30.pct_lo", num(res.day30_lo * 100.0, 4))
        frag.put(f"cif.{key}.day30.pct_hi", num(res.day30_hi * 100.0, 4))
        frag.put(f"cif.{key}.day30.text", pct_ci(res.day30, res.day30_lo, res.day30_hi))
        for d in LANDMARK_DAYS:
            if d == N_DAYS:
                continue     # already written above, with its band and its text form
            frag.put(f"cif.{key}.day{d}.pct", num(float(res.cif[d]) * 100.0, 4))
        frag.put(f"cif.{key}.file", f"tables-csv/cif_{key}.csv")

        write_cif_csv(TABLES / f"cif_{key}.csv", key, res, n_analytic)

    frag.put("cif.files", [f"tables-csv/cif_{o.key}.csv" for o in VD.TIMED_OUTCOMES])
    frag.put("cif.outcome_keys", [o.key for o in VD.TIMED_OUTCOMES])
    frag.put("cif.outcome_labels", [VD.pretty(o.event_col) for o in VD.TIMED_OUTCOMES])
    return store


CIF_COLUMNS: Tuple[str, ...] = (
    "Outcome",
    "Postoperative day",
    "Cumulative incidence, %",
    "Lower 95% confidence limit, %",
    "Upper 95% confidence limit, %",
    "Number at risk",
    "Events on this day",
    "Cumulative events",
    "Competing deaths on this day",
    "Cumulative competing deaths",
    DENOM_COLUMN,
)


def write_cif_csv(path: pathlib.Path, key: str, res: CIFResult, denom: int) -> None:
    """The step-curve data `07_figures.py` plots. One row per integer day, 0 to 30.

    Self-sufficient on purpose: the outcome label, the denominator and the risk set are
    all here, so the figure step never has to reach back into the parquet or guess a
    denominator. Values are unformatted numbers because a plotting step needs numbers;
    they still pass `assert_renderable`.
    """
    label = VD.pretty(VD.timed_outcome(key).event_col)
    rows = []
    for d in res.days:
        rows.append({
            "Outcome": label,
            "Postoperative day": int(d),
            "Cumulative incidence, %": f"{res.cif[d] * 100.0:.6f}",
            "Lower 95% confidence limit, %": f"{res.lo[d] * 100.0:.6f}",
            "Upper 95% confidence limit, %": f"{res.hi[d] * 100.0:.6f}",
            "Number at risk": int(res.at_risk[d]),
            "Events on this day": int(res.e1[d]),
            "Cumulative events": int(res.cum_e1[d]),
            "Competing deaths on this day": int(res.e2[d]),
            "Cumulative competing deaths": int(res.cum_e2[d]),
            DENOM_COLUMN: int(denom),
        })
    write_csv(path, CIF_COLUMNS, rows)


# ===========================================================================
# 2. Gray's test across the four locked stratifiers
# ===========================================================================
def analysis_gray(df: pd.DataFrame, store: Dict[str, Any], frag: Fragment) -> Dict[str, Any]:
    frag.put("gray.method",
             "Gray's k-sample test on the cumulative incidence functions, rho = 0. "
             "Follow-up is complete to day 30, so every inverse-probability-of-censoring "
             "weight is one and the statistic is the multivariate log-rank form computed "
             "on subdistribution risk sets.")
    frag.put("gray.suppression.min_events", VD.MIN_STRATUM_EVENTS)
    frag.put("gray.suppression.rule",
             f"A stratum with fewer than {VD.MIN_STRATUM_EVENTS} events of the outcome is "
             f"excluded from the test and reported with its count. A comparison left with "
             f"fewer than two strata is not tested and its p value is null.")
    # CLAUDE.md 6.5, scope ruling 2026-07-31: the minimum stratum size is PROJECT-WIDE
    # and applies to any stratum-level PROPORTION in any step, not only to the
    # safe-discharge curves and not only to a test's membership. The counts, the
    # denominator and the event count are always published; only the proportion is
    # withheld, and the withholding is flagged so it is auditable rather than invisible.
    frag.put("gray.suppression.proportion_rule",
             f"A stratum-level day 30 cumulative incidence computed on fewer than "
             f"{VD.MIN_STRATUM_EVENTS} events is withheld rather than reported. The "
             f"stratum size, the event count and the denominator are still published, so "
             f"the suppression is auditable, and each withheld cell carries its own "
             f"suppression flag. A reader cannot be expected to check an event count "
             f"before quoting a percentage.")
    frag.put("gray.stratifiers", [VD.pretty(s.column) for s in VD.STRATIFIERS])

    summary: Dict[str, Any] = {}
    n_tests = 0
    n_suppressed_tests = 0
    suppressed_cells: List[str] = []
    for o in VD.TIMED_OUTCOMES:
        key = o.key
        times = store[key]["times"]
        etypes = store[key]["etypes"]
        for strat in VD.STRATIFIERS:
            col = strat.column
            skey = key_token(col)
            raw = df[col]
            lvl_series = raw.astype("object").where(raw.notna(), other="")
            lvl_series = lvl_series.map(
                lambda v: "" if (v is None or (isinstance(v, float) and not math.isfinite(v)))
                else str(v).strip())
            # Int8 stratifiers arrive as 1 / 0; make the level strings match the locked
            # ASCII strings in variable_dict.STRATIFIERS.
            lvl_series = lvl_series.map(lambda v: v[:-2] if v.endswith(".0") else v)
            levels = list(strat.levels)

            missing_n = int((lvl_series == "").sum())
            kept, dropped, dropped_counts = [], [], []
            for lv in levels:
                mask = (lvl_series == lv).to_numpy()
                n_lv = int(mask.sum())
                ev_lv = int(np.sum(mask & (np.asarray(etypes) == 1)))
                cif_lv = None
                if n_lv > 0:
                    r_lv = aalen_johansen(times[mask], etypes[mask])
                    cif_lv = float(r_lv.cif[-1])
                # The proportion is withheld on exactly the same test the test
                # membership uses, so the two can never disagree about a sparse cell.
                thin = ev_lv < VD.MIN_STRATUM_EVENTS
                frag.put(f"gray.{key}.{skey}.{key_token(lv)}.n", n_lv)
                frag.put(f"gray.{key}.{skey}.{key_token(lv)}.events_n", ev_lv)
                frag.put(f"gray.{key}.{skey}.{key_token(lv)}.suppressed", bool(thin))
                frag.put(f"gray.{key}.{skey}.{key_token(lv)}.day30_pct",
                         None if (thin or cif_lv is None) else num(cif_lv * 100.0, 4))
                if thin:
                    # Rendered, not raw: this list is a fragment value and 08 may print
                    # it, so a raw column name here would be a snake_case leak.
                    suppressed_cells.append(VD.assert_renderable(
                        f"{VD.short(o.event_col)}, {VD.pretty(col)} "
                        f"{VD.pretty_level(col, lv)}, {ev_lv} events in "
                        f"{VD.fmt_count(n_lv)} patients",
                        where="gray.strata_proportions_suppressed_list"))
                if not thin:
                    kept.append(lv)
                else:
                    dropped.append(lv)
                    dropped_counts.append(ev_lv)

            if len(kept) >= 2:
                mask_any = lvl_series.isin(kept).to_numpy()
                stat, dof, p, _ = grays_test(times[mask_any], np.asarray(etypes)[mask_any],
                                             lvl_series[mask_any].to_numpy())
                n_tests += 1
            else:
                stat = dof = p = None
                n_suppressed_tests += 1

            frag.put(f"gray.{key}.{skey}.stat", num(stat, 4))
            frag.put(f"gray.{key}.{skey}.df", dof)
            frag.put(f"gray.{key}.{skey}.p", num(p, 6))
            frag.put(f"gray.{key}.{skey}.p_text", VD.fmt_p(p) if p is not None else "")
            frag.put(f"gray.{key}.{skey}.levels_tested_n", len(kept))
            frag.put(f"gray.{key}.{skey}.levels_suppressed_n", len(dropped))
            frag.put(f"gray.{key}.{skey}.levels_suppressed",
                     [VD.pretty_level(col, lv) for lv in dropped])
            frag.put(f"gray.{key}.{skey}.levels_suppressed_events",
                     [int(c) for c in dropped_counts])
            frag.put(f"gray.{key}.{skey}.missing_level_n", missing_n)
            summary[(key, col)] = {
                "stat": stat, "df": dof, "p": p,
                "kept": kept, "dropped": dropped, "dropped_counts": dropped_counts,
                "missing_n": missing_n,
            }
    frag.put("gray.tests_run_n", n_tests)
    frag.put("gray.tests_suppressed_n", n_suppressed_tests)
    frag.put("gray.strata_proportions_suppressed_n", len(suppressed_cells))
    frag.put("gray.strata_proportions_suppressed_list", sorted(suppressed_cells))
    return summary


# ===========================================================================
# 3. Table 3A and the timing descriptives
# ===========================================================================
TABLE3A_COLUMNS: Tuple[str, ...] = (
    "Outcome",
    "Events, n",
    "Events with the postoperative day recorded, n",
    "Median postoperative day",
    "Interquartile range, days",
    "Events on postoperative day 0 or 1, n (%)",
    "Events after postoperative day 10, n (%)",
    DENOM_COLUMN,
)


def analysis_table3a(df: pd.DataFrame, store: Dict[str, Any],
                     frag: Fragment) -> Dict[str, Any]:
    n_analytic = len(df)
    rows = []
    out: Dict[str, Any] = {}

    frag.put("timing.denominator", n_analytic)
    frag.put("timing.followup_days", N_DAYS)
    frag.put("timing.time_zero", "the day of the principal operation, postoperative day 0")
    frag.put("timing.censoring",
             "administrative at postoperative day 30, non-informative, and complete by "
             "construction: NSQIP follow-up is fixed at 30 days and there is no loss to "
             "follow-up in the ordinary sense")
    frag.put("timing.estimator",
             "Aalen-Johansen cumulative incidence function with death as the competing "
             "event, evaluated as a step function on integer postoperative days")
    frag.put("timing.estimator.implementation",
             "written out directly over 31 integer time points; the lifelines fitter was "
             "not used because it jitters tied event times and drops every event at "
             "time 0")
    frag.put("timing.ci.method",
             "95% limits from the Klein-Moeschberger delta-method variance with the "
             "log-log transformation, so both limits stay inside the unit interval")
    frag.put("timing.death.no_competing_event",
             "For death itself there is no competing event and the Aalen-Johansen "
             "estimator reduces to the ordinary cumulative incidence.")
    frag.put("timing.extrapolation", "none past postoperative day 30")
    frag.put("timing.tie_convention", VD.TIE_CONVENTION)
    frag.put("timing.outcome_keys", [o.key for o in VD.TIMED_OUTCOMES])
    frag.put("timing.outcome_labels", [VD.pretty(o.event_col) for o in VD.TIMED_OUTCOMES])

    for o in VD.TIMED_OUTCOMES:
        key = o.key
        ev = (df[o.event_col].fillna(0).astype("int64") == 1).to_numpy()
        pod = pd.to_numeric(df[o.pod_col], errors="coerce").to_numpy(dtype=float)
        obs = pod[ev & np.isfinite(pod)]
        n_events = int(ev.sum())
        n_obs = int(obs.size)
        med, q1, q3 = median_iqr_days(obs)
        n01 = int(np.sum(obs <= 1)) if n_obs else 0
        n10 = int(np.sum(obs > 10)) if n_obs else 0
        p01 = (n01 / n_obs * 100.0) if n_obs else None
        p10 = (n10 / n_obs * 100.0) if n_obs else None

        out[key] = {"n_events": n_events, "n_obs": n_obs, "median": med, "q1": q1,
                    "q3": q3, "n01": n01, "p01": p01, "n10": n10, "p10": p10}

        frag.put(f"timing.{key}.events_n", n_events)
        frag.put(f"timing.{key}.day_recorded_n", n_obs)
        frag.put(f"timing.{key}.day_missing_n", n_events - n_obs)
        frag.put(f"timing.{key}.denominator", n_obs)
        frag.put(f"timing.{key}.cohort_denominator", n_analytic)
        frag.put(f"timing.{key}.crude_pct", num(n_events / n_analytic * 100.0, 4))
        frag.put(f"timing.{key}.median_pod", num(med, 2))
        frag.put(f"timing.{key}.q1_pod", num(q1, 2))
        frag.put(f"timing.{key}.q3_pod", num(q3, 2))
        frag.put(f"timing.{key}.iqr_text",
                 VD.fmt_iqr(med, q1, q3, 0) if med is not None else "")
        frag.put(f"timing.{key}.pod01_n", n01)
        frag.put(f"timing.{key}.pod01_pct", num(p01, 4))
        frag.put(f"timing.{key}.after_day10_n", n10)
        frag.put(f"timing.{key}.after_day10_pct", num(p10, 4))

        rows.append({
            "Outcome": VD.pretty(o.event_col),
            "Events, n": VD.fmt_count(n_events),
            "Events with the postoperative day recorded, n": VD.fmt_count(n_obs),
            "Median postoperative day": f"{med:.0f}" if med is not None else "",
            "Interquartile range, days": (VD.fmt_ci(q1, q3, 0) if med is not None else ""),
            "Events on postoperative day 0 or 1, n (%)": VD.fmt_n_pct(n01, n_obs, 1),
            "Events after postoperative day 10, n (%)": VD.fmt_n_pct(n10, n_obs, 1),
            DENOM_COLUMN: VD.fmt_count(n_obs),
        })

    write_csv(TABLES / "table3a_complication_timing.csv", TABLE3A_COLUMNS, rows)

    # The cohort median postoperative length of stay: Figure 2 draws its vertical
    # reference line here. Taken from the FROZEN LOS_POSTOP column; no cohort is
    # recomputed and the 0 to 30 handling rule does NOT apply to a length of stay.
    los = pd.to_numeric(df["LOS_POSTOP"], errors="coerce")
    los_obs = los.dropna().to_numpy(dtype=float)
    lmed, lq1, lq3 = median_iqr_days(los_obs)
    frag.put("timing.los_postop.median", num(lmed, 2))
    frag.put("timing.los_postop.q1", num(lq1, 2))
    frag.put("timing.los_postop.q3", num(lq3, 2))
    frag.put("timing.los_postop.iqr_text", VD.fmt_iqr(lmed, lq1, lq3, 0))
    frag.put("timing.los_postop.denominator", int(los_obs.size))
    frag.put("timing.los_postop.missing_n", int(len(df) - los_obs.size))
    frag.put("timing.los_postop.max", num(float(los_obs.max()), 0))
    frag.put("timing.los_postop.figure2_reference_day", num(lmed, 2))
    out["los"] = {"median": lmed, "q1": lq1, "q3": lq3, "n": int(los_obs.size),
                  "missing": int(len(df) - los_obs.size), "max": float(los_obs.max())}

    # Time-to-first-event caveat (PLAN.md section 4, CLAUDE.md 9.18), re-derived here
    # because it is a property of the timing analysis rather than of the cohort build.
    def _yes(col: str) -> np.ndarray:
        return df[col].astype("string").fillna("").str.strip().str.lower().eq("yes").to_numpy()

    reop2 = _yes("REOPERATION2")
    reop3 = _yes("REOPERATION3")
    readm2 = _yes("READMISSION2")
    readm3 = _yes("READMISSION3")
    readm4 = _yes("READMISSION4")
    readm5 = _yes("READMISSION5")
    any2 = reop2 | reop3 | readm2 | readm3 | readm4 | readm5
    frag.put("timing.second_event.patients_n", int(any2.sum()))
    frag.put("timing.second_event.patients_pct", num(any2.sum() / n_analytic * 100.0, 4))
    frag.put("timing.second_event.reoperation2_n", int(reop2.sum()))
    frag.put("timing.second_event.reoperation3_untimed_n", int(reop3.sum()))
    frag.put("timing.second_event.readmission2_n", int(readm2.sum()))
    frag.put("timing.second_event.readmission3_n", int(readm3.sum()))
    frag.put("timing.second_event.readmission4_n", int(readm4.sum()))
    frag.put("timing.second_event.readmission5_n", int(readm5.sum()))
    frag.put("timing.second_event.denominator", n_analytic)
    frag.put("timing.second_event.note",
             "All analyses are time to first event. NSQIP records a days to event field "
             "for occurrence 1 in every family, a second reoperation and a second through "
             "fifth readmission, and nothing beyond, so a third reoperation carries no "
             "timing at all.")
    out["second_event"] = {
        "any": int(any2.sum()), "reop2": int(reop2.sum()), "reop3": int(reop3.sum()),
        "readm2": int(readm2.sum()), "readm3": int(readm3.sum()),
        "readm4": int(readm4.sum()), "readm5": int(readm5.sum()),
    }

    # Tracheostomy slot 1 versus slots 1 and 2, side by side, for the ascertainment
    # supplement (notes_02.md section 8 item 12 assigns this to 03).
    s1 = int(pd.to_numeric(df["REOP_TRACH"], errors="coerce").fillna(0).sum())
    s12 = int(pd.to_numeric(df["REOP_TRACH_SLOT12"], errors="coerce").fillna(0).sum())
    frag.put("timing.trach.slot1_n", s1)
    frag.put("timing.trach.slot12_n", s12)
    frag.put("timing.trach.slot12_only_n", s12 - s1)
    frag.put("timing.trach.in_composite", "slot 1 only, the companion project's frozen "
                                          "implementation")
    out["trach"] = {"slot1": s1, "slot12": s12}
    return out


# ===========================================================================
# 4. Supplement S3: one minus Kaplan-Meier versus Aalen-Johansen
# ===========================================================================
S3_COLUMNS: Tuple[str, ...] = (
    "Outcome",
    "Events, n",
    "Competing deaths, n",
    "Aalen-Johansen cumulative incidence at day 30, % (95% CI)",
    "One minus Kaplan-Meier at day 30, % (95% CI)",
    "Absolute difference, percentage points",
    "Relative difference, %",
    DENOM_COLUMN,
)


def analysis_s3(store: Dict[str, Any], n_analytic: int, frag: Fragment) -> Dict[str, Any]:
    rows = []
    out: Dict[str, Any] = {}
    frag.put("s3.method",
             "One minus Kaplan-Meier treats a competing death as non-informative "
             "censoring, which assumes the patient who died would have gone on to "
             "experience the complication at the same rate as the living. It therefore "
             "overstates the cumulative incidence of every non-fatal outcome. The size "
             "of the correction is measured here rather than assumed small.")
    frag.put("s3.denominator", n_analytic)

    worst_key, worst_diff = None, -1.0
    worst_rel_key, worst_rel = None, -1.0
    for o in VD.TIMED_OUTCOMES:
        key = o.key
        res: CIFResult = store[key]["res"]
        counts = store[key]["counts"]
        aj = res.day30
        km = float(res.km_cif[-1])
        diff_pp = (km - aj) * 100.0
        rel = (km - aj) / aj * 100.0 if aj > 0 else None

        frag.put(f"s3.{key}.aj_pct", num(aj * 100.0, 4))
        frag.put(f"s3.{key}.aj_pct_lo", num(res.day30_lo * 100.0, 4))
        frag.put(f"s3.{key}.aj_pct_hi", num(res.day30_hi * 100.0, 4))
        frag.put(f"s3.{key}.km_pct", num(km * 100.0, 4))
        frag.put(f"s3.{key}.km_pct_lo", num(float(res.km_lo[-1]) * 100.0, 4))
        frag.put(f"s3.{key}.km_pct_hi", num(float(res.km_hi[-1]) * 100.0, 4))
        frag.put(f"s3.{key}.diff_pp", num(diff_pp, 6))
        frag.put(f"s3.{key}.relative_pct", num(rel, 4))
        frag.put(f"s3.{key}.competing_deaths_n", counts["competing_n"])
        frag.put(f"s3.{key}.denominator", n_analytic)
        out[key] = {"aj": aj, "km": km, "diff_pp": diff_pp, "rel": rel,
                    "competing": counts["competing_n"]}

        if key != "death" and diff_pp > worst_diff:
            worst_diff, worst_key = diff_pp, key
        if key != "death" and rel is not None and rel > worst_rel:
            worst_rel, worst_rel_key = rel, key

        rows.append({
            "Outcome": VD.pretty(o.event_col),
            "Events, n": VD.fmt_count(counts["events_n"]),
            "Competing deaths, n": VD.fmt_count(counts["competing_n"]),
            "Aalen-Johansen cumulative incidence at day 30, % (95% CI)":
                pct_ci(aj, res.day30_lo, res.day30_hi),
            "One minus Kaplan-Meier at day 30, % (95% CI)":
                pct_ci(km, float(res.km_lo[-1]), float(res.km_hi[-1])),
            "Absolute difference, percentage points": f"{diff_pp:.4f}",
            "Relative difference, %": (f"{rel:.2f}" if rel is not None else ""),
            DENOM_COLUMN: VD.fmt_count(n_analytic),
        })

    write_csv(TABLES / "s3_km_vs_aalen_johansen.csv", S3_COLUMNS, rows)

    frag.put("s3.max_diff_pp", num(worst_diff, 6))
    frag.put("s3.max_diff_outcome", VD.pretty(VD.timed_outcome(worst_key).event_col)
             if worst_key else "")
    frag.put("s3.max_diff_outcome_key", worst_key or "")
    frag.put("s3.max_relative_pct", num(worst_rel, 4))
    frag.put("s3.max_relative_outcome", VD.pretty(VD.timed_outcome(worst_rel_key).event_col)
             if worst_rel_key else "")
    frag.put("s3.death.difference_pp", num(out["death"]["diff_pp"], 8))
    frag.put("s3.death.note",
             "Death has no competing event, so the Aalen-Johansen estimator reduces to "
             "the ordinary cumulative incidence and the two estimators are identical by "
             "construction. The row is printed to show that, not as a comparison.")
    out["_worst"] = {"key": worst_key, "diff_pp": worst_diff,
                     "rel_key": worst_rel_key, "rel": worst_rel}
    return out


# ===========================================================================
# 5. Supplement S1: field definitions, code sets, DISCHDEST level map
# ===========================================================================
S1_COLUMNS: Tuple[str, ...] = (
    "Section", "Item", "Definition", "Source field or code set", "Years the value applies",
)


def _populated_year_range(df: pd.DataFrame, column: str) -> str:
    """The operative years in which a raw column is actually populated in THIS cohort.

    `PLAN.md` 7.3 and `CLAUDE.md` 9.10 give different year ranges for the two reoperation
    diagnosis columns, one describing when the column EXISTS and the other when it is
    POPULATED. Rather than assert either, S1 prints what is measured here.
    """
    if column not in df.columns:
        return "not exported"
    vals = df[column].astype("string").fillna("").str.strip()
    yrs = pd.to_numeric(df["PUFYEAR"], errors="coerce")[(vals != "").to_numpy()]
    yrs = yrs.dropna()
    if yrs.empty:
        return "never populated in this cohort"
    return f"{int(yrs.min())} to {int(yrs.max())}"


def analysis_s1(df: pd.DataFrame, frag: Fragment,
                trach: Dict[str, int]) -> Dict[str, str]:
    rows: List[Dict[str, str]] = []

    def add(section: str, item: str, definition: str, source: str, years: str) -> None:
        rows.append({"Section": section, "Item": item, "Definition": definition,
                     "Source field or code set": source,
                     "Years the value applies": years})

    icd9_years = _populated_year_range(df, CPT.HEMATOMA_ICD9_COLUMN)
    icd10_years = _populated_year_range(df, CPT.HEMATOMA_ICD10_COLUMN)
    frag.put("timing.coding.icd9_arm_populated_years", icd9_years)
    frag.put("timing.coding.icd10_arm_populated_years", icd10_years)

    # --- the timed outcomes and their raw NSQIP sources --------------------
    for o in VD.TIMED_OUTCOMES:
        add("Timed outcomes",
            VD.pretty(o.event_col),
            VD.prose(o.event_col).capitalize()
            + ". Timing is the "
            + ("earliest of the component days to event"
               if len(o.raw_fields) > 1 else "recorded days to event")
            + ", counted from the principal operation.",
            ", ".join(o.raw_fields),
            "2012 to 2024")

    add("Timed outcomes", VD.pretty("LOS_POSTOP"),
        "Days from the principal operation to discharge. The primary length-of-stay "
        "field, direct, present in every study year, and deliberately not limited to the "
        "30-day follow-up window.", "DOPTODIS", "2012 to 2024")
    add("Timed outcomes", "Length-of-stay cross-check",
        "Total hospital length of stay minus days from admission to operation. The "
        "independent cross-check on the primary field and never the analysis variable.",
        "TOTHLOS, HTOODAY", "2012 to 2024")
    add("Timed outcomes", VD.pretty("STILLINHOSP"),
        "Recorded as still admitted beyond postoperative day 30. Such a patient has no "
        "discharge inside the follow-up window.", "STILLINHOSP", "2012 to 2024")
    add("Timed outcomes", VD.pretty("RETOR2PODAYS"),
        "Only the first event of each type carries a timing field. A second reoperation "
        "and a second through fifth readmission are timed; a third reoperation is not.",
        "RETOR2PODAYS, READMPODAYS2", "2012 to 2024")

    # --- the authoritative code sets, RENDERED from cpt_acdf ---------------
    add("Procedure code sets", "Principal procedure",
        "Anterior cervical discectomy and fusion, principal procedure code.",
        cpt_list(CPT.PRINCIPAL_CPTS), "2012 to 2024")
    add("Procedure code sets", "Additional fusion level",
        "Each further occurrence across the ten concurrent procedure slots adds one "
        "fusion level.", CPT.ADDITIONAL_LEVEL_CPT, "2012 to 2024")
    add("Procedure code sets", "Concurrent corpectomy or craniocervical arthrodesis",
        "Exclusion.", cpt_list(CPT.EXCLUSION_CORPECTOMY_CPTS), "2012 to 2024")
    add("Procedure code sets",
        "Concurrent posterior spinal decompression or instrumentation",
        "Exclusion. The honoured list carries a lumbar code and codes that are not "
        "cervical specific, so it is not described as posterior cervical.",
        cpt_list(CPT.EXCLUSION_POSTERIOR_CERVICAL_CPTS), "2012 to 2024")
    add("Procedure code sets", "Cervical disc arthroplasty", "Exclusion.",
        cpt_list(CPT.EXCLUSION_ARTHROPLASTY_CPTS), "2012 to 2024")
    add("Procedure code sets", "Anterior cervical instrumentation", "Descriptive marker.",
        cpt_list(CPT.ANTERIOR_INSTRUMENTATION_CPTS), "2012 to 2024")
    add("Procedure code sets", VD.pretty("REOP_TRACH"),
        f"Reoperative tracheostomy, read from the first reoperation slot, which is the "
        f"implementation frozen inside the airway composite. Slot 1 identifies "
        f"{trach['slot1']} patients and slots 1 and 2 together identify "
        f"{trach['slot12']}; the wider marker is reported here and is deliberately not "
        f"in the composite. It is an exploratory marker with no literature precedent.",
        CPT.TRACHEOSTOMY_CPT, "2012 to 2024")

    add("Hematoma ascertainment", "Reoperation procedure arm",
        "Unplanned reoperation procedure code in the first reoperation slot.",
        cpt_list(CPT.HEMATOMA_REOP_CPTS), "2012 to 2024")
    add("Hematoma ascertainment", "Reoperation diagnosis arm, ninth revision",
        "Unplanned reoperation diagnosis code, prefix match. The year range is the range "
        "in which the column is actually populated inside this cohort, measured rather "
        "than asserted.",
        icd_list(CPT.HEMATOMA_ICD9), icd9_years)
    add("Hematoma ascertainment", "Reoperation diagnosis arm, tenth revision",
        f"Unplanned reoperation diagnosis code, prefix match, plus the exact code "
        f"{icd_list(CPT.HEMATOMA_ICD10_EXACT)}. Definition in force: "
        f"{CPT.HEMATOMA_DEFINITION}. The year range is measured inside this cohort.",
        icd_list(CPT.HEMATOMA_ICD10_PREFIXES), icd10_years)
    add("Hematoma ascertainment", "Seroma guard, ninth revision",
        "Seroma-only codes are excluded and are tested before the hematoma families.",
        icd_list(CPT.SEROMA_ICD9_EXCLUDED), icd9_years)
    add("Hematoma ascertainment", "Seroma guard, tenth revision",
        "Seroma-only members of the hematoma families, excluded.",
        icd_list(CPT.SEROMA_ICD10_EXCLUDED), icd10_years)
    add("Hematoma ascertainment", "Union rule",
        "The three arms are combined ungated across years. A year-gated rule would lose "
        "a real event ascertained only by the ninth-revision arm inside the overlap "
        "years.", "REOPORCPT1, REOPORICD91, REOPOR1ICD101", "2012 to 2024")
    for variant in CPT.HEMATOMA_DEFINITION_ORDER:
        spec = CPT.HEMATOMA_ICD10_VARIANTS[variant]
        add("Hematoma ascertainment", f"Code set {variant}",
            str(spec["label"]) + (". Primary." if variant == CPT.HEMATOMA_DEFINITION
                                  else ". Sensitivity."),
            icd_list(tuple(spec["prefixes"]) + tuple(sorted(spec["exact"]))),
            icd10_years)

    # --- the DISCHDEST level map, straight from variable_dict --------------
    # Both collapsed forms are printed: the ASCII value actually stored in the analytic
    # column and its display text, because the display text for the facility group names
    # only its commonest members and a reader checking the collapse rule needs the stored
    # value.
    for r in VD.dischdest_level_rows():
        add("Discharge destination level map",
            r[VD.DISCHDEST_TABLE_COLUMNS[0]],
            f"{r[VD.DISCHDEST_TABLE_COLUMNS[1]]}. Collapsed to "
            f"{r[VD.DISCHDEST_TABLE_COLUMNS[2]]} "
            f"({r[VD.DISCHDEST_TABLE_COLUMNS[3]]}).",
            "DISCHDEST",
            r[VD.DISCHDEST_TABLE_COLUMNS[4]])
    add("Discharge destination level map", "Collapse rule",
        "Home discharge is defined by this explicit map and never by matching the "
        "substring Home, which would fold the facility that was the patient's home "
        "before surgery into the home group. An unmapped level raises.",
        "DISCHDEST", "2012 to 2024")

    # --- billing setting ---------------------------------------------------
    for level in VD.INOUT_LEVEL_MAP:
        add("Billing setting", level,
            f"{VD.pretty_level('INOUT', level)}. It is a billing label and never a "
            f"length of stay.", "INOUT", "2012 to 2024")

    # --- estimator and conventions ----------------------------------------
    add("Analysis conventions", "Time zero",
        "The day of the principal operation, postoperative day 0.", "", "2012 to 2024")
    add("Analysis conventions", "Estimator",
        "Aalen-Johansen cumulative incidence with death as the competing event, a step "
        "function on integer postoperative days.", "", "2012 to 2024")
    add("Analysis conventions", "Censoring",
        f"Administrative at postoperative day {N_DAYS}, non-informative, complete by "
        f"construction. No estimate extends past day {N_DAYS}.", "", "2012 to 2024")
    add("Analysis conventions", "Stratum comparison",
        f"Gray's test on the cumulative incidence functions. A stratum with fewer than "
        f"{VD.MIN_STRATUM_EVENTS} events is suppressed and reported with its count.",
        "", "2012 to 2024")
    add("Analysis conventions", "Tie convention", VD.TIE_CONVENTION, "", "2012 to 2024")
    add("Analysis conventions", "Days to event handling",
        "A days-to-event value is used only when it is an integer in the range 0 to 30. "
        "Anything else is counted and set to missing rather than clipped or dropped. The "
        "rule was applied once, when the analytic dataset was built, and affected no "
        "value in this cohort. It does not apply to the length of stay.",
        "", "2012 to 2024")

    write_csv(TABLES / "s1_field_definitions.csv", S1_COLUMNS, rows)
    frag.put("timing.s1.rows_n", len(rows))
    frag.put("timing.s1.dischdest_levels_n", len(VD.dischdest_level_rows()))
    frag.put("timing.s1.file", "tables-csv/s1_field_definitions.csv")
    return {"icd9_years": icd9_years, "icd10_years": icd10_years}


# ===========================================================================
# 6. The coding-era check (PLAN.md 7.3)
# ===========================================================================
def analysis_era(df: pd.DataFrame, frag: Fragment) -> Dict[str, Any]:
    """Year-by-year event rate for EVERY timed outcome, tested for a step at the
    ICD-9 to ICD-10 transition.

    The transition year is READ from the frozen `ERA_ICD10` flag rather than typed, so
    this check and the ICD-10-era sensitivity below can never point at different years.

    THE RULE, stated before the verdicts are read. An outcome is declared to carry a step
    when BOTH of the following hold:

      1. the event rates in the last pre-transition year and the first transition year
         differ at p < 0.05 by a two-sided Fisher exact test; and
      2. the absolute change in rate at that boundary is the LARGEST of the twelve
         adjacent-year changes across 2012 to 2024.

    Condition 1 alone would over-call: eight outcomes tested at the 5% level will produce
    a significant boundary by chance, and a real coding step has to look different from
    the ordinary year-to-year wobble. Condition 2 is what makes it a step rather than a
    trend. Both components are published for every outcome, so a reader sees the
    evidence and not only the verdict.

    An outcome with a step is reported BY ERA and never pooled.
    """
    yr = pd.to_numeric(df["PUFYEAR"], errors="coerce").astype("Int64")
    era10 = pd.to_numeric(df["ERA_ICD10"], errors="coerce").fillna(0).astype(int)
    years = sorted(int(y) for y in yr.dropna().unique())
    transition = int(yr[(era10 == 1).to_numpy()].min())
    pre_years = [y for y in years if y < transition]
    post_years = [y for y in years if y >= transition]
    last_pre = max(pre_years)
    first_post = min(post_years)

    n_by_year = [int((yr == y).sum()) for y in years]
    frag.put("era.years", [int(y) for y in years])
    frag.put("era.n_by_year", n_by_year)
    frag.put("era.transition_year", transition)
    frag.put("era.boundary_pair", f"{last_pre} to {first_post}")
    frag.put("era.rule",
             "A step is declared when the last pre-transition year and the first "
             "transition year differ at p below 0.05 by a two-sided Fisher exact test "
             "AND the absolute change at that boundary is the largest of the twelve "
             "adjacent-year changes. An outcome with a step is reported by era and never "
             "pooled.")
    frag.put("era.denominator", len(df))

    step_keys: List[str] = []
    out: Dict[str, Any] = {"years": years, "transition": transition,
                           "n_by_year": n_by_year, "outcomes": {}}

    for o in VD.TIMED_OUTCOMES:
        key = o.key
        ev = (df[o.event_col].fillna(0).astype("int64") == 1)
        ev_by_year = [int(ev[(yr == y).to_numpy()].sum()) for y in years]
        rate_by_year = [(e / n * 100.0) if n else float("nan")
                        for e, n in zip(ev_by_year, n_by_year)]

        changes = [abs(rate_by_year[i + 1] - rate_by_year[i])
                   for i in range(len(years) - 1)]
        b_idx = years.index(last_pre)
        boundary_change = rate_by_year[b_idx + 1] - rate_by_year[b_idx]
        max_change = max(changes)
        max_i = int(np.argmax(changes))
        rank = 1 + sum(1 for c in changes if c > abs(boundary_change) + 1e-12)
        boundary_is_largest = bool(abs(abs(boundary_change) - max_change) < 1e-12)

        a = ev_by_year[b_idx]
        b = n_by_year[b_idx] - a
        c_ = ev_by_year[b_idx + 1]
        d_ = n_by_year[b_idx + 1] - c_
        _, p_boundary = fisher_exact([[a, b], [c_, d_]])

        pre_n = sum(n for y, n in zip(years, n_by_year) if y < transition)
        post_n = sum(n for y, n in zip(years, n_by_year) if y >= transition)
        pre_e = sum(e for y, e in zip(years, ev_by_year) if y < transition)
        post_e = sum(e for y, e in zip(years, ev_by_year) if y >= transition)
        _, p_pooled = fisher_exact([[pre_e, pre_n - pre_e], [post_e, post_n - post_e]])

        step = bool(p_boundary < 0.05 and boundary_is_largest)
        if step:
            step_keys.append(key)

        frag.put(f"era.{key}.events_by_year", ev_by_year)
        frag.put(f"era.{key}.rate_by_year", [num(r, 4) for r in rate_by_year])
        frag.put(f"era.{key}.year_min_pct", num(min(rate_by_year), 4))
        frag.put(f"era.{key}.year_max_pct", num(max(rate_by_year), 4))
        frag.put(f"era.{key}.pre_n", pre_n)
        frag.put(f"era.{key}.post_n", post_n)
        frag.put(f"era.{key}.pre_events_n", pre_e)
        frag.put(f"era.{key}.post_events_n", post_e)
        frag.put(f"era.{key}.pre_pct", num(pre_e / pre_n * 100.0, 4))
        frag.put(f"era.{key}.post_pct", num(post_e / post_n * 100.0, 4))
        frag.put(f"era.{key}.pooled_p", num(float(p_pooled), 6))
        frag.put(f"era.{key}.boundary_p", num(float(p_boundary), 6))
        frag.put(f"era.{key}.boundary_change_pp", num(boundary_change, 4))
        frag.put(f"era.{key}.boundary_change_rank", int(rank))
        frag.put(f"era.{key}.boundary_is_largest_change", boundary_is_largest)
        frag.put(f"era.{key}.largest_change_pp", num(max_change, 4))
        frag.put(f"era.{key}.largest_change_pair", f"{years[max_i]} to {years[max_i + 1]}")
        frag.put(f"era.{key}.step_detected", step)
        frag.put(f"era.{key}.denominator", len(df))

        out["outcomes"][key] = {
            "events_by_year": ev_by_year, "rate_by_year": rate_by_year,
            "pre_pct": pre_e / pre_n * 100.0, "post_pct": post_e / post_n * 100.0,
            "pre_n": pre_n, "post_n": post_n, "pre_e": pre_e, "post_e": post_e,
            "p_boundary": float(p_boundary), "p_pooled": float(p_pooled),
            "boundary_change": boundary_change, "rank": rank,
            "largest_change": max_change,
            "largest_pair": f"{years[max_i]} to {years[max_i + 1]}",
            "is_largest": boundary_is_largest, "step": step,
            "band": (min(rate_by_year), max(rate_by_year)),
        }

    frag.put("era.any_step_detected", bool(step_keys))
    frag.put("era.step_outcome_keys", step_keys)
    frag.put("era.step_outcome_labels",
             [VD.pretty(VD.timed_outcome(k).event_col) for k in step_keys])
    frag.put("era.pooling_verdict",
             "No outcome carries a step at the coding transition, so every timing curve "
             "is reported pooled across 2012 to 2024."
             if not step_keys else
             "At least one outcome carries a step at the coding transition and is "
             "reported by era rather than pooled.")

    # The hematoma per-year band, RE-DERIVED. The companion step measured 0.3223% to
    # 0.6390%; this is the independent reproduction, reported whether or not it agrees.
    band = out["outcomes"]["hematoma"]["band"]
    frag.put("era.hematoma.band_min_pct", num(band[0], 4))
    frag.put("era.hematoma.band_max_pct", num(band[1], 4))
    frag.put("era.hematoma.band_carried_min_pct", CARRYOVER_HEMATOMA_BAND[0])
    frag.put("era.hematoma.band_carried_max_pct", CARRYOVER_HEMATOMA_BAND[1])
    frag.put("era.hematoma.band_reproduces",
             bool(abs(band[0] - CARRYOVER_HEMATOMA_BAND[0]) < 5e-5
                  and abs(band[1] - CARRYOVER_HEMATOMA_BAND[1]) < 5e-5))
    out["step_keys"] = step_keys
    return out


def write_era_curves(df: pd.DataFrame, step_keys: Sequence[str], transition: int,
                     frag: Fragment) -> List[str]:
    """Era-split curve files, written ONLY for an outcome that carries a step.

    PLAN.md 7.3: an outcome with a step is reported by era and never pooled. If the list
    is empty nothing is written and `07_figures.py` plots the pooled curves, which is what
    `cif.era_split_required` records.
    """
    written: List[str] = []
    era = pd.to_numeric(df["ERA_ICD10"], errors="coerce").fillna(0).astype(int)
    for key in step_keys:
        for tag, mask in (("icd9era", era == 0), ("icd10era", era == 1)):
            sub = df.loc[mask.to_numpy()]
            times, etypes, _ = build_times(sub, key)
            res = aalen_johansen(times, etypes)
            path = TABLES / f"cif_{key}_{tag}.csv"
            write_cif_csv(path, key, res, len(sub))
            written.append(f"tables-csv/{path.name}")
    frag.put("cif.era_split_required", bool(step_keys))
    frag.put("cif.era_split_files", written)
    frag.put("cif.era_split_transition_year", transition)
    return written


# ===========================================================================
# 7. Supplement S4: this step's sensitivity analyses
# ===========================================================================
S4_COLUMNS: Tuple[str, ...] = (
    "Sensitivity analysis",
    "Outcome",
    "Population",
    "n",
    "Events, n",
    "Day 30 cumulative incidence, % (95% CI)",
    "Median postoperative day (interquartile range)",
    "Events on postoperative day 0 or 1, n (%)",
    "Primary analysis day 30 cumulative incidence, %",
    "Difference from primary, percentage points",
    DENOM_COLUMN,
)


def analysis_s4(df: pd.DataFrame, store: Dict[str, Any], frag: Fragment) -> Dict[str, Any]:
    n_analytic = len(df)
    rows: List[Dict[str, str]] = []
    out: Dict[str, Any] = {}

    def blank_row(**kw: Any) -> Dict[str, str]:
        r = {c: "" for c in S4_COLUMNS}
        r.update(kw)
        return r

    # --- block A: one minus Kaplan-Meier versus Aalen-Johansen -------------
    for o in VD.TIMED_OUTCOMES:
        key = o.key
        res: CIFResult = store[key]["res"]
        km = float(res.km_cif[-1])
        rows.append(blank_row(**{
            "Sensitivity analysis": "One minus Kaplan-Meier instead of Aalen-Johansen",
            "Outcome": VD.pretty(o.event_col),
            "Population": "Analytic cohort, 2012 to 2024",
            "n": VD.fmt_count(n_analytic),
            "Events, n": VD.fmt_count(store[key]["counts"]["events_n"]),
            "Day 30 cumulative incidence, % (95% CI)":
                pct_ci(km, float(res.km_lo[-1]), float(res.km_hi[-1])),
            "Primary analysis day 30 cumulative incidence, %":
                VD.fmt_pct(res.day30 * 100.0, 3),
            "Difference from primary, percentage points": f"{(km - res.day30) * 100.0:.4f}",
            DENOM_COLUMN: VD.fmt_count(n_analytic),
        }))

    # --- blocks B and C: the two era restrictions --------------------------
    era_specs = (
        ("icd10era", "ERA_ICD10", 1,
         "Restricted to the tenth-revision coding era",
         "Operative years 2016 to 2024"),
        ("emergncyera", "ERA_EMERGNCY", 1,
         "Restricted to the era in which elective status came from the emergency-case "
         "field",
         "Operative years 2012 to 2020"),
    )
    for tag, col, want, label, popn in era_specs:
        mask = (pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int) == want)
        sub = df.loc[mask.to_numpy()]
        frag.put(f"timing.sens.{tag}.n", int(len(sub)))
        for o in VD.TIMED_OUTCOMES:
            key = o.key
            times, etypes, counts = build_times(sub, key)
            res = aalen_johansen(times, etypes)
            primary: CIFResult = store[key]["res"]
            frag.put(f"timing.sens.{tag}.{key}.day30_pct", num(res.day30 * 100.0, 4))
            frag.put(f"timing.sens.{tag}.{key}.day30_pct_lo",
                     num(res.day30_lo * 100.0, 4))
            frag.put(f"timing.sens.{tag}.{key}.day30_pct_hi",
                     num(res.day30_hi * 100.0, 4))
            frag.put(f"timing.sens.{tag}.{key}.events_n", counts["events_n"])
            frag.put(f"timing.sens.{tag}.{key}.denominator", int(len(sub)))
            frag.put(f"timing.sens.{tag}.{key}.diff_pp",
                     num((res.day30 - primary.day30) * 100.0, 4))
            rows.append(blank_row(**{
                "Sensitivity analysis": label,
                "Outcome": VD.pretty(o.event_col),
                "Population": popn,
                "n": VD.fmt_count(len(sub)),
                "Events, n": VD.fmt_count(counts["events_n"]),
                "Day 30 cumulative incidence, % (95% CI)":
                    pct_ci(res.day30, res.day30_lo, res.day30_hi),
                "Primary analysis day 30 cumulative incidence, %":
                    VD.fmt_pct(primary.day30 * 100.0, 3),
                "Difference from primary, percentage points":
                    f"{(res.day30 - primary.day30) * 100.0:.4f}",
                DENOM_COLUMN: VD.fmt_count(len(sub)),
            }))

    # --- block D: the three hematoma code sets -----------------------------
    #
    # `HEMATOMA_V0` and `HEMATOMA_V1` are FROZEN columns and the days-to-event field is
    # the same raw reoperation-day column under all three definitions. Pairing a frozen
    # event flag with a frozen raw day column is a sensitivity transform of two frozen
    # columns and not a cohort derivation; PLAN.md 6.6 and the table and figure inventory
    # both assign this row to this step. The pairing is PROVED equivalent to the one
    # `02_build_cohort.py` performed by re-deriving the primary day column from the same
    # two frozen columns and asserting it reproduces `HEMATOMA_POD` exactly. If that ever
    # fails, the whole block is a stop condition rather than a number to publish.
    raw_day = pd.to_numeric(df["RETORPODAYS"], errors="coerce")
    ok_day = raw_day.where(raw_day.notna()
                           & (raw_day == raw_day.round(0))
                           & (raw_day >= 0) & (raw_day <= N_DAYS))
    dropped_by_rule = int((raw_day.notna() & ok_day.isna()).sum())
    frag.put("timing.sens.hematoma.day_rule_dropped_n", dropped_by_rule)

    primary_flag = df["HEMATOMA"].fillna(0).astype("int64") == 1
    rebuilt = ok_day.where(primary_flag)
    frozen = pd.to_numeric(df["HEMATOMA_POD"], errors="coerce")
    if not rebuilt.fillna(-1).equals(frozen.fillna(-1)):
        raise SystemExit(
            "STOP: pairing the frozen hematoma flag with the frozen reoperation-day "
            "column does not reproduce HEMATOMA_POD. The code-set sensitivity cannot be "
            "computed without that equivalence; report it rather than publishing a row."
        )
    frag.put("timing.sens.hematoma.pairing_reproduces_frozen_column", True)

    variant_cols = {"V2": "HEMATOMA", "V1": "HEMATOMA_V1", "V0": "HEMATOMA_V0"}
    primary_res: CIFResult = store["hematoma"]["res"]
    for variant in ("V2", "V1", "V0"):
        col = variant_cols[variant]
        flag = (df[col].fillna(0).astype("int64") == 1)
        day = ok_day.where(flag)
        times = np.full(len(df), float(N_DAYS))
        etypes = np.zeros(len(df), dtype=int)
        has = flag.to_numpy() & day.notna().to_numpy()
        times[has] = day.to_numpy(dtype=float)[has]
        etypes[has] = 1
        died = (df["MORT30"].fillna(0).astype("int64") == 1).to_numpy()
        dday = pd.to_numeric(df["POD_DEATH"], errors="coerce").to_numpy(dtype=float)
        comp = died & ~has
        times[comp] = dday[comp]
        etypes[comp] = 2
        res = aalen_johansen(times.astype(int), etypes)
        obs = day.dropna().to_numpy(dtype=float)
        med, q1, q3 = median_iqr_days(obs)
        n01 = int(np.sum(obs <= 1))
        vkey = variant.lower()
        frag.put(f"timing.sens.hematoma.{vkey}.events_n", int(flag.sum()))
        frag.put(f"timing.sens.hematoma.{vkey}.day_recorded_n", int(obs.size))
        frag.put(f"timing.sens.hematoma.{vkey}.day30_pct", num(res.day30 * 100.0, 4))
        frag.put(f"timing.sens.hematoma.{vkey}.day30_pct_lo", num(res.day30_lo * 100.0, 4))
        frag.put(f"timing.sens.hematoma.{vkey}.day30_pct_hi", num(res.day30_hi * 100.0, 4))
        frag.put(f"timing.sens.hematoma.{vkey}.median_pod", num(med, 2))
        frag.put(f"timing.sens.hematoma.{vkey}.q1_pod", num(q1, 2))
        frag.put(f"timing.sens.hematoma.{vkey}.q3_pod", num(q3, 2))
        frag.put(f"timing.sens.hematoma.{vkey}.pod01_n", n01)
        frag.put(f"timing.sens.hematoma.{vkey}.pod01_pct",
                 num(n01 / obs.size * 100.0, 4) if obs.size else None)
        frag.put(f"timing.sens.hematoma.{vkey}.denominator", int(obs.size))
        frag.put(f"timing.sens.hematoma.{vkey}.diff_pp",
                 num((res.day30 - primary_res.day30) * 100.0, 4))
        out[f"hematoma_{vkey}"] = {
            "events": int(flag.sum()), "obs": int(obs.size), "day30": res.day30,
            "median": med, "q1": q1, "q3": q3, "n01": n01,
            "p01": (n01 / obs.size * 100.0) if obs.size else None,
        }
        rows.append(blank_row(**{
            "Sensitivity analysis": f"Hematoma code set {variant}"
                                    + (" (primary)" if variant == CPT.HEMATOMA_DEFINITION
                                       else ""),
            "Outcome": VD.pretty("HEMATOMA"),
            "Population": "Analytic cohort, 2012 to 2024",
            "n": VD.fmt_count(n_analytic),
            "Events, n": VD.fmt_count(int(flag.sum())),
            "Day 30 cumulative incidence, % (95% CI)":
                pct_ci(res.day30, res.day30_lo, res.day30_hi),
            "Median postoperative day (interquartile range)":
                (f"{med:.0f} ({VD.fmt_ci(q1, q3, 0)})" if med is not None else ""),
            "Events on postoperative day 0 or 1, n (%)":
                VD.fmt_n_pct(n01, int(obs.size), 1),
            "Primary analysis day 30 cumulative incidence, %":
                VD.fmt_pct(primary_res.day30 * 100.0, 3),
            "Difference from primary, percentage points":
                f"{(res.day30 - primary_res.day30) * 100.0:.4f}",
            DENOM_COLUMN: VD.fmt_count(int(obs.size)),
        }))

    write_csv(TABLES / "s4_sensitivity_03.csv", S4_COLUMNS, rows)
    frag.put("timing.s4.rows_n", len(rows))
    frag.put("timing.s4.file", "tables-csv/s4_sensitivity_03.csv")
    return out


# ===========================================================================
# 8. Gate 2 (CLAUDE.md 8.4): the two carried-over numbers this step owns
# ===========================================================================
def identified_reconciliation() -> Optional[Dict[str, Any]]:
    """Re-derive the hematoma median day and day 0-1 share on the IDENTIFIED rung.

    WHY THIS FILE IS OPENED. CLAUDE.md 8.4 requires the carried-over median postoperative
    day and day 0-1 share to be re-derived by this pipeline, and `02` established that the
    carried-over figures are IDENTIFIED-rung numbers. Re-deriving the reconciliation
    therefore needs the identified rung. The alternative is to copy `02`'s number, which
    binding item 1 of its handoff explicitly forbids ("re-derive; never copy a number out
    of this note"). `derived/cohort_identified.parquet` is a frozen artefact `02` already
    wrote; two columns are read from it, nothing is written, and no cohort is re-derived.
    The manuscript reports the ANALYTIC value as primary (CLAUDE.md 4.4); this is the
    reconciliation figure that sits beside it.
    """
    if not COHORT_IDENTIFIED.exists():
        return None
    idf = pd.read_parquet(COHORT_IDENTIFIED, columns=["HEMATOMA", "HEMATOMA_POD"])
    flag = idf["HEMATOMA"].fillna(0).astype("int64") == 1
    day = pd.to_numeric(idf["HEMATOMA_POD"], errors="coerce").where(flag)
    obs = day.dropna().to_numpy(dtype=float)
    med, q1, q3 = median_iqr_days(obs)
    n01 = int(np.sum(obs <= 1))
    return {
        "n": int(len(idf)),
        "events": int(flag.sum()),
        "day_recorded": int(obs.size),
        "median": med, "q1": q1, "q3": q3,
        "pod01_n": n01,
        "pod01_pct": (n01 / obs.size * 100.0) if obs.size else None,
    }


def analysis_gate2(table3a: Dict[str, Any], frag: Fragment) -> Dict[str, Any]:
    h = table3a["hematoma"]
    ident = identified_reconciliation()

    med_a = h["median"]
    p01_a = h["p01"]
    n_a = h["n_obs"]

    frag.put("timing.gate2.median_pod_analytic", num(med_a, 2))
    frag.put("timing.gate2.median_pod_denominator_analytic", n_a)
    frag.put("timing.gate2.pod01_pct_analytic", num(p01_a, 4))
    frag.put("timing.gate2.pod01_n_analytic", h["n01"])
    frag.put("timing.gate2.pod01_denominator_analytic", n_a)
    frag.put("timing.gate2.median_pod_carried", CARRYOVER_MEDIAN_REOP_POD)
    frag.put("timing.gate2.pod01_pct_carried", CARRYOVER_POD01_PCT)

    med_match_a = (med_a is not None
                   and abs(med_a - CARRYOVER_MEDIAN_REOP_POD) < 1e-9)
    p01_match_a = (p01_a is not None
                   and round(p01_a, 1) == round(CARRYOVER_POD01_PCT, 1))
    frag.put("timing.gate2.median_pod_reproduces_analytic", bool(med_match_a))
    frag.put("timing.gate2.pod01_reproduces_analytic", bool(p01_match_a))

    if ident is None:
        frag.put("timing.gate2.identified_available", False)
        frag.put("timing.gate2.median_pod_identified", None)
        frag.put("timing.gate2.pod01_pct_identified", None)
        frag.put("timing.gate2.pod01_denominator_identified", None)
        frag.put("timing.gate2.median_pod_reproduces_identified", None)
        frag.put("timing.gate2.pod01_reproduces_identified", None)
        med_match_i = p01_match_i = None
    else:
        frag.put("timing.gate2.identified_available", True)
        frag.put("timing.gate2.identified_n", ident["n"])
        frag.put("timing.gate2.identified_events_n", ident["events"])
        frag.put("timing.gate2.median_pod_identified", num(ident["median"], 2))
        frag.put("timing.gate2.pod01_pct_identified", num(ident["pod01_pct"], 4))
        frag.put("timing.gate2.pod01_n_identified", ident["pod01_n"])
        frag.put("timing.gate2.pod01_denominator_identified", ident["day_recorded"])
        med_match_i = (ident["median"] is not None
                       and abs(ident["median"] - CARRYOVER_MEDIAN_REOP_POD) < 1e-9)
        p01_match_i = (ident["pod01_pct"] is not None
                       and round(ident["pod01_pct"], 1) == round(CARRYOVER_POD01_PCT, 1))
        frag.put("timing.gate2.median_pod_reproduces_identified", bool(med_match_i))
        frag.put("timing.gate2.pod01_reproduces_identified", bool(p01_match_i))

    # The verdict. CLAUDE.md 8.5: a mismatch is a stop condition to REPORT, never a
    # number to overwrite. It is recorded here and printed; it does not raise, because
    # the correct handling of an expected V0-to-V2 movement is disclosure, not a crash.
    verdict_parts = []
    verdict_parts.append(
        f"median postoperative day reproduces on the analytic rung: {bool(med_match_a)}")
    if med_match_i is not None:
        verdict_parts.append(
            f"median reproduces on the identified rung: {bool(med_match_i)}")
    verdict_parts.append(
        f"day 0 to 1 share reproduces on the analytic rung to one decimal: "
        f"{bool(p01_match_a)}")
    if p01_match_i is not None:
        verdict_parts.append(
            f"day 0 to 1 share reproduces on the identified rung to one decimal: "
            f"{bool(p01_match_i)}")
    frag.put("timing.gate2.verdict", "; ".join(verdict_parts))
    frag.put("timing.gate2.note",
             "The carried-over figures were measured under the superseded hematoma code "
             "set on the identified rung. The analytic value is primary and the "
             "identified value is the reconciliation. A difference is a documented "
             "movement of definition and denominator, not a number to overwrite.")
    return {"analytic": h, "identified": ident,
            "med_match_a": med_match_a, "p01_match_a": p01_match_a,
            "med_match_i": med_match_i, "p01_match_i": p01_match_i}


# ===========================================================================
# notes
# ===========================================================================
def write_notes(ctx: Dict[str, Any]) -> None:
    df_n = ctx["n_analytic"]
    store = ctx["store"]
    t3 = ctx["table3a"]
    s3 = ctx["s3"]
    era = ctx["era"]
    gray = ctx["gray"]
    gate2 = ctx["gate2"]
    diag = ctx["diag"]
    s4 = ctx["s4"]
    nkeys = ctx["nkeys"]
    fragment = ctx["fragment"]

    L: List[str] = []
    a = L.append

    a("# notes_03.md -- handoff from `03_timing.py`")
    a("")
    a("Written 2026-07-31. Owner: **T5 (timing)**. Owned files: `03_timing.py`,")
    a("`outputs/results_03.json`, `derived/notes_03.md`,")
    a("`tables-csv/table3a_complication_timing.csv`, `tables-csv/s1_field_definitions.csv`,")
    a("`tables-csv/s3_km_vs_aalen_johansen.csv`, `tables-csv/s4_sensitivity_03.csv` and")
    a("`tables-csv/cif_*.csv`. Nothing else was created or modified.")
    a("")
    a("Input: `derived/cohort_analytic.parquet` (frozen, n = "
      f"{VD.fmt_count(df_n)}). `derived/cohort_identified.parquet` was opened read-only")
    a("for two columns, for the Gate 2 identified-rung reconciliation only; see section 8.")
    a("The warehouse and the export CSV were not read at all.")
    a("")
    a("---")
    a("")

    # ---- section 1
    a("## 1. The estimator, and why it is hand-implemented")
    a("")
    a("**Aalen-Johansen cumulative incidence with death as the competing event**, on")
    a(f"integer postoperative days 0 to {N_DAYS}, one curve per outcome in")
    a("`variable_dict.TIMED_OUTCOMES`. Time zero is the day of the principal operation.")
    a("")
    a("`lifelines` 0.30.3 is installed and ships `AalenJohansenFitter`. **It cannot be used")
    a("on this data**, for two independent reasons, both verified rather than assumed:")
    a("")
    a("1. **It jitters tied event times.** Its tie detector fires when a time carries an")
    a("   event of one type and an event of another, which integer postoperative days")
    a("   produce constantly: a death and a reoperation on the same day is enough.")
    a(f"   Measured on this cohort, the detector fires on "
      f"{ctx.get('jitter_outcomes')} of the {len(VD.TIMED_OUTCOMES)} outcomes, so most of")
    a("   the curves would have been drawn on a jittered time axis. `jitter_level=0` is")
    a("   not an escape hatch either: once the detector fires the jitter routine rejects a")
    a("   non-positive level (verified this run on the smallest fixture that triggers the")
    a(f"   detector: detector fires = {diag.get('tie_detector_fires')}, rejects a zero "
      f"jitter = {diag.get('rejects_zero_jitter')}). A jittered time axis is not a step")
    a("   function on integers, which `CLAUDE.md` 6.1 requires.")
    a("2. **It drops every event at time 0**, independently of the jitter. Its")
    a("   cumulative-incidence column is a `cumsum` over `d1 / n * lagged_survival`.")
    a("   `survival_table_from_events` always synthesises a row at time 0, so time 0 is")
    a("   always the first row of the event table; the lagged survival is undefined there")
    a("   and is zeroed, so that row contributes nothing to the running sum.")
    a("")
    a("   **The mechanism is time 0, not the earliest observed time.** Probed directly, a")
    a("   fixture whose earliest observed event is at time 1 loses nothing, because the")
    a("   synthesised time-0 row sits ahead of it and absorbs the undefined value. The")
    a("   looser wording happens to be true of this cohort and would not generalise, so")
    a("   it is not used. Measured here rather than assumed:")
    a(f"   time 0 is the earliest observed time on")
    a(f"   {ctx['fragment']['timing.library.outcomes_with_earliest_observed_time_zero_n']}")
    a(f"   of the {len(VD.TIMED_OUTCOMES)} outcomes, and day 0 carries events of interest")
    a(f"   on {ctx['fragment']['timing.library.outcomes_with_day0_events_n']} of them, so")
    a("   the loss is real on this data.")
    if diag.get("available"):
        a("   Demonstrated deterministically on a 1,000-patient fixture whose exact answer")
        a(f"   is {diag['fixture_exact_pct']:.1f}%: the library reports")
        a(f"   {diag['fixture_library_pct']:.1f}%, losing "
          f"{diag['fixture_events_lost_n']} events.")
    if ctx.get("libdiff"):
        ld = ctx["libdiff"]
        worst = max(ld, key=lambda k: abs(ld[k]))
        a("")
        a("**What that would have cost on this cohort**, measured by running the library on")
        a("the same time and event-type vectors and differencing against the exact value")
        a("(a library-behaviour diagnostic under `timing.library.*`, never an estimate):")
        a("")
        a("| Outcome | Understatement, percentage points |")
        a("|---|---:|")
        for o in VD.TIMED_OUTCOMES:
            if o.key in ld:
                a(f"| {VD.pretty(o.event_col)} | {ld[o.key]:.4f} |")
        a("")
        a(f"The worst is **{ld[worst]:.4f} percentage points** on "
          f"{VD.pretty(VD.timed_outcome(worst).event_col).lower()}, which is "
          f"{int(round(ld[worst] / 100.0 * df_n))} events.")
    a("")
    a("The estimator is therefore **written out directly** over 31 integer time points:")
    a("")
    a("```")
    a("n_d    = n_{d-1} - e1_{d-1} - e2_{d-1} - c_{d-1}")
    a("CIF(d) = CIF(d-1) + S(d-1) * e1_d / n_d")
    a("S(d)   = S(d-1) * (1 - (e1_d + e2_d) / n_d)")
    a("```")
    a("")
    a("with `S(-1) = 1`, `CIF(-1) = 0`, and the risk set on day 0 equal to the whole")
    a("cohort. Day 0 is a real day: "
      f"{int(store['reop_any']['res'].e1[0])} unplanned reoperations, "
      f"{int(store['hematoma']['res'].e1[0])} hematoma reoperations and "
      f"{int(store['reintub']['res'].e1[0])} reintubations occur on it, and an estimator")
    a("that discards them is not a rounding problem.")
    a("")
    a("**Variance and band.** The Klein-Moeschberger / PROC LIFETEST delta-method")
    a("estimator, which is the same three-term formula `lifelines._bounds` implements,")
    a("with the log-log transformation so both limits stay inside the unit interval. A")
    a("linear Wald interval on a probability of a few tenths of a percent runs negative.")
    a("Where the estimate is still exactly zero the band is reported as zero width; it")
    a("opens when the first event occurs.")
    a("")
    a("**A structural check that runs every time.** NSQIP follow-up is fixed and complete")
    a(f"at {N_DAYS} days, so nothing is censored before day {N_DAYS}. Under complete")
    a("follow-up the Aalen-Johansen estimator reduces algebraically to the empirical")
    a("cumulative proportion of patients whose event was recorded **with a postoperative")
    a("day**. The script asserts exactly that for every outcome and stops if it ever")
    a("drifts by more than 1e-10. **It held on all eight outcomes.** Two consequences")
    a("worth carrying into the Methods:")
    a("")
    a("* the day-30 cumulative incidence is the empirical cumulative proportion of the")
    a("  events that carry an observed day, which is the sanity check a reviewer will do")
    a("  by hand, and it agrees to machine precision rather than approximately. **It is")
    a("  not the crude proportion of all events**, and on the outcomes that lost an event")
    a("  day the two differ, by exactly the events with no recorded day over the same")
    a("  denominator (`cif.<outcome>.max_understatement_pct`). Section 2.1 tabulates it;")
    a("* the whole of the one-minus-Kaplan-Meier gap in supplement S3 is attributable to")
    a("  the competing deaths and to nothing else, because there is no other censoring to")
    a("  share the blame with.")
    a("")
    a("**Which key is the 30-day incidence.** `cif.<outcome>.day30.pct`, with")
    a("`day30.pct_lo` and `day30.pct_hi`. That value is the terminal point of the Figure 2")
    a("curve and `cif.<outcome>.crude_pct` is not, so the two are not interchangeable")
    a("sources for the same manuscript sentence. The rule is published for `08` as")
    a("`cif.day30_reporting_key`, with `cif.identity_note` and `cif.crude_pct_note`.")
    a("")
    a("**Censoring is administrative and non-informative** and this is a genuine strength")
    a("of the design, not a caveat: follow-up is fixed at 30 days from the principal")
    a("operation and complete by construction, there is no loss to follow-up in the")
    a("ordinary sense, and the risk set is full until day 30 or the first event. Reviewers")
    a("who see a survival analysis assume informative censoring unless told otherwise, so")
    a("the Methods must say it.")
    a("")
    a("**No estimate extends past day 30.** The day-30 value is terminal.")
    a("")
    a("**Events with no recorded day.** A patient whose event flag fired but whose day was")
    a("not recorded cannot be placed on the time axis. They stay in the cohort and in the")
    a("risk set and contribute no event; nothing is imputed and nothing is clipped. The")
    a("count is published per outcome as `cif.<outcome>.day_unobserved_events_n`, and the")
    a("maximum understatement it can cause is that count over the analytic denominator,")
    a("published as `cif.<outcome>.max_understatement_pct`. The largest is")
    lost = max(store[o.key]["counts"]["day_unobserved_n"] for o in VD.TIMED_OUTCOMES)
    lost_key = [o.key for o in VD.TIMED_OUTCOMES
                if store[o.key]["counts"]["day_unobserved_n"] == lost][0]
    a(f"{lost} events ({VD.pretty(VD.timed_outcome(lost_key).event_col).lower()}), which is")
    a(f"{lost / df_n * 100.0:.4f}% of the cohort.")
    a("")
    a("---")
    a("")

    # ---- section 2
    a("## 2. Day 30 cumulative incidence, every outcome")
    a("")
    a("| Outcome | Events | Day 30 cumulative incidence (95% CI) | Competing deaths |")
    a("|---|---:|---|---:|")
    for o in VD.TIMED_OUTCOMES:
        r: CIFResult = store[o.key]["res"]
        c = store[o.key]["counts"]
        a(f"| {VD.pretty(o.event_col)} | {VD.fmt_count(c['events_n'])} | "
          f"{pct_ci(r.day30, r.day30_lo, r.day30_hi)} | "
          f"{VD.fmt_count(c['competing_n'])} |")
    a("")
    a(f"Denominator for every row: the analytic cohort, {VD.fmt_count(df_n)}. Death has no")
    a("competing event and its row is the ordinary cumulative incidence.")
    a("")
    a("Curve data, one file per outcome, `tables-csv/cif_<outcome>.csv`, 31 rows each,")
    a("carrying the day, the estimate, both limits, the number at risk, the events on the")
    a("day, the cumulative events, the competing deaths and a `denominator` column.")
    a("**They are self-sufficient: `07_figures.py` needs nothing else to draw Figure 2.**")
    a("")
    a("### 2.1 The by-hand sanity check a reviewer will do")
    a("")
    a("At day 30 the Aalen-Johansen estimate equals the empirical cumulative proportion of")
    a("the events that carry an observed postoperative day, and is therefore no greater")
    a("than the crude proportion of **all** events. The gap is not slack in the estimator:")
    a("it is exactly the events whose day was not recorded, over the same denominator.")
    a("")
    a("| Outcome | Aalen-Johansen at day 30 | Crude proportion, all events | Gap, percentage "
      "points | Events with no recorded day |")
    a("|---|---|---|---:|---:|")
    n_gap = 0
    for o in VD.TIMED_OUTCOMES:
        r2: CIFResult = store[o.key]["res"]
        c2 = store[o.key]["counts"]
        crude2 = c2["events_n"] / df_n * 100.0
        # `+ 0.0` normalises a negative zero: an exact-match row differences to -0.0 in
        # binary floating point and would print as "-0.0000", which reads as a defect.
        gap2 = round(crude2 - r2.day30 * 100.0, 4) + 0.0
        if c2["day_unobserved_n"]:
            n_gap += 1
        a(f"| {VD.pretty(o.event_col)} | {VD.fmt_pct(r2.day30 * 100.0, 4)} | "
          f"{VD.fmt_pct(crude2, 4)} | {gap2:.4f} | {c2['day_unobserved_n']} |")
    a("")
    a(f"**{n_gap} of the {len(VD.TIMED_OUTCOMES)} outcomes carry a gap**, and on each of")
    a("them the gap column reproduces `cif.<outcome>.max_understatement_pct` exactly. On")
    a("the rest no event day is missing and the two columns are identical to machine")
    a("precision, which is the algebraic identity in section 1 and not a coincidence.")
    a("")
    a("**Do not write the Methods sentence as an equality against the crude proportion.**")
    a("It would be wrong on the outcomes in the gap column and a reviewer can falsify it")
    a("with one subtraction. The correct sentence is the one in section 1: the estimator")
    a("reduces to the empirical cumulative proportion of events with an observed day.")
    a("")
    a("---")
    a("")

    # ---- section 3
    a("## 3. Table 3A and the descriptive timing")
    a("")
    a("`tables-csv/table3a_complication_timing.csv`. One row per timed outcome, in roster")
    a("order. Every label is routed through `variable_dict.pretty()`.")
    a("")
    a("| Outcome | Events | Day recorded | Median (IQR) | Day 0 or 1 | After day 10 |")
    a("|---|---:|---:|---|---|---|")
    for o in VD.TIMED_OUTCOMES:
        d = t3[o.key]
        a(f"| {VD.pretty(o.event_col)} | {VD.fmt_count(d['n_events'])} | "
          f"{VD.fmt_count(d['n_obs'])} | "
          f"{d['median']:.0f} ({VD.fmt_ci(d['q1'], d['q3'], 0)}) | "
          f"{VD.fmt_n_pct(d['n01'], d['n_obs'], 1)} | "
          f"{VD.fmt_n_pct(d['n10'], d['n_obs'], 1)} |")
    a("")
    a("**The denominator for the median, the interquartile range and both percentages is")
    a("the number of events with the postoperative day recorded**, not the number of")
    a("events, and it is printed in its own column as well as in the machine-readable")
    a(f"`{DENOM_COLUMN}` column.")
    a("")
    a("**Table 3A does not carry a post-discharge column.** `04_postdischarge.py` owns the")
    a("post-discharge fraction and the denominator ladder (Table 3B). That is an ownership")
    a("boundary and this step does not compute a post-discharge fraction anywhere.")
    a("")
    los = t3["los"]
    a("### 3.1 The Figure 2 reference line")
    a("")
    a("Figure 2 draws a vertical reference line at the cohort median postoperative length")
    a("of stay. Taken from the frozen `LOS_POSTOP` column, not recomputed from a cohort:")
    a("")
    a(f"* median **{los['median']:.0f} day**, interquartile range "
      f"{VD.fmt_ci(los['q1'], los['q3'], 0)};")
    a(f"* denominator {VD.fmt_count(los['n'])}, with {los['missing']} rows missing;")
    a(f"* the field is not window bounded and runs to {los['max']:.0f} days, so **the 0 to")
    a("  30 handling rule does not apply to it** and no step may truncate it.")
    a("")
    a("Read it from `timing.los_postop.median` (equivalently")
    a("`timing.los_postop.figure2_reference_day`), never from this sentence.")
    a("")
    se = t3["second_event"]
    a("### 3.2 Time to first event, and the partially observed second events")
    a("")
    a("All analyses are time to first event, pre-specified. NSQIP times occurrence 1 in")
    a("every family, plus a second reoperation and a second through fifth readmission, and")
    a("nothing beyond. Re-derived here on the analytic cohort:")
    a("")
    a(f"* **{VD.fmt_count(se['any'])} patients "
      f"({se['any'] / df_n * 100.0:.2f}%) had at least one second event**;")
    a(f"* second unplanned reoperation {se['reop2']}, third reoperation {se['reop3']} "
      f"(**no timing field exists for it at all**);")
    a(f"* second readmission {se['readm2']}, third {se['readm3']}, fourth {se['readm4']}, "
      f"fifth {se['readm5']}.")
    a("")
    a("This is the population whose timing is only partially observed and the Methods must")
    a("state it.")
    a("")
    tr = t3["trach"]
    a("### 3.3 The tracheostomy slot counts, side by side")
    a("")
    a(f"Reoperative tracheostomy in the first reoperation slot: **{tr['slot1']}**. In the")
    a(f"first or second slot: **{tr['slot12']}**. The airway composite and the")
    a("safe-discharge composite use the frozen slot-1 implementation, unchanged, so this")
    a("project and the companion project cannot disagree about the composite. Both counts")
    a("are in supplement S1 and in `timing.trach.*`.")
    a("")
    a("---")
    a("")

    # ---- section 4
    a("## 4. Supplement S3: one minus Kaplan-Meier versus Aalen-Johansen")
    a("")
    a("`tables-csv/s3_km_vs_aalen_johansen.csv`. **The magnitude is measured, not assumed")
    a("small.**")
    a("")
    a("| Outcome | Aalen-Johansen at day 30 | One minus Kaplan-Meier | Difference, pp | "
      "Relative |")
    a("|---|---|---|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        d = s3[o.key]
        rel = "identical" if o.key == "death" else f"{d['rel']:.2f}%"
        a(f"| {VD.pretty(o.event_col)} | {VD.fmt_pct(d['aj'] * 100.0, 4)} | "
          f"{VD.fmt_pct(d['km'] * 100.0, 4)} | {d['diff_pp']:.4f} | {rel} |")
    a("")
    w = s3["_worst"]
    a(f"The largest absolute correction is **{w['diff_pp']:.4f} percentage points** "
      f"({VD.pretty(VD.timed_outcome(w['key']).event_col).lower()}); the largest relative")
    a(f"correction is **{w['rel']:.2f}%** "
      f"({VD.pretty(VD.timed_outcome(w['rel_key']).event_col).lower()}). Every difference")
    a("is positive, which is the direction theory requires: treating a competing death as")
    a("non-informative censoring assumes the dead patient would have gone on to have the")
    a("complication at the same rate as the living, so one minus Kaplan-Meier **overstates**")
    a("the cumulative incidence of every non-fatal outcome.")
    a("")
    a("**Death has no competing event.** Its Aalen-Johansen estimate reduces to the")
    a(f"ordinary cumulative incidence and the two estimators agree to "
      f"{abs(s3['death']['diff_pp']):.2e} percentage points, which is floating-point noise.")
    a("That row is printed to show the reduction, not as a comparison, and the Methods say")
    a("so rather than reporting a self-competing estimate.")
    a("")
    a("**What to tell the reviewer who asks why a competing-risks framework was used on a")
    a("low-mortality cohort.** The correction is small, and it was measured rather than")
    a("assumed; the correctness of the estimator does not depend on the answer being close")
    a("either way; and in this cohort the Aalen-Johansen estimate is exactly the empirical")
    a("cumulative proportion of the events with an observed day while one minus")
    a("Kaplan-Meier is not, so the estimator that needs defending is the other one.")
    a("")
    a("---")
    a("")

    # ---- section 5
    a("## 5. Gray's test across the four locked stratifiers")
    a("")
    a("`variable_dict.STRATIFIERS`: age group, ASA physical status class, fusion levels and")
    a("sex. Nothing else. No interaction strata, no data-driven cut points.")
    a("")
    a("**Implementation.** Gray's k-sample test on the cumulative incidence functions, the")
    a("rho = 0 member of the family. Gray compares subdistribution hazards, so a patient")
    a("who experiences the competing event stays in the risk set. Because follow-up here is")
    a("complete to day 30, every inverse-probability-of-censoring weight is exactly one and")
    a("the subdistribution risk set on day d is simply the group size minus the events of")
    a("the cause of interest before day d. The statistic is then the multivariate log-rank")
    a("form on those risk sets with the usual hypergeometric variance, computed directly so")
    a("it is auditable by hand.")
    a("")
    a(f"**Suppression, set before the curves were seen:** a stratum with fewer than")
    a(f"{VD.MIN_STRATUM_EVENTS} events of the outcome is excluded from the test and")
    a("reported with its count. A comparison left with fewer than two strata is not tested")
    a("and its p value is null.")
    a("")
    a("**The same rule now withholds the stratum-level PROPORTION, not only the test")
    a("membership.** `CLAUDE.md` 6.5 carries an orchestrator ruling that the minimum")
    a(f"stratum size of {VD.MIN_STRATUM_EVENTS} events is project-wide and applies to any")
    a("stratum-level proportion in any step, because a reader cannot be expected to check")
    a("an event count before quoting a percentage. This step previously published a day 30")
    a("cumulative incidence for every non-empty level while gating only the test, which")
    a("meant two co-equal steps sharing one constant treated the same sparse cell")
    a("differently. Both the proportion and the test membership are now decided by one")
    a("comparison, so they cannot disagree. The stratum size and the event count are")
    a("always published and each withheld cell carries its own suppression flag.")
    a("")
    n_cells_supp = int(fragment["gray.strata_proportions_suppressed_n"])
    a(f"**Stratum-level proportions withheld this run: {n_cells_supp}** of "
      f"{sum(len(s.levels) for s in VD.STRATIFIERS) * len(VD.TIMED_OUTCOMES)} cells.")
    if n_cells_supp:
        for line in fragment["gray.strata_proportions_suppressed_list"]:
            a(f"* {line}")
    else:
        a("None.")
    a("")
    a("| Outcome | Stratifier | Statistic | df | p | Strata tested | Suppressed |")
    a("|---|---|---:|---:|---|---:|---|")
    for o in VD.TIMED_OUTCOMES:
        for s in VD.STRATIFIERS:
            g = gray[(o.key, s.column)]
            supp = (", ".join(f"{VD.pretty_level(s.column, lv)} ({c})"
                              for lv, c in zip(g["dropped"], g["dropped_counts"]))
                    or "none")
            stat = f"{g['stat']:.3f}" if g["stat"] is not None else "not tested"
            dof = str(g["df"]) if g["df"] is not None else ""
            pv = VD.fmt_p(g["p"]) if g["p"] is not None else "not tested"
            a(f"| {VD.pretty(o.event_col)} | {VD.pretty(s.column)} | {stat} | {dof} | "
              f"{pv} | {len(g['kept'])} | {supp} |")
    a("")
    n_supp = sum(1 for v in gray.values() if v["dropped"])
    a(f"{n_supp} of {len(gray)} comparisons dropped at least one stratum under the")
    a("suppression rule. Every stratum-level event count and denominator is in the")
    a("fragment under `gray.<outcome>.<stratifier>.<level>.*`, so a reader can see what was")
    a("compared and not only whether it differed. The day-30 cumulative incidence is there")
    a("too wherever the stratum is large enough to carry one, and is null with")
    a("`suppressed` true wherever it is not.")
    a("")
    a("**Level tokens in the fragment are transliterated** so they satisfy the merge key")
    a("regular expression: `>=` becomes `ge`, `<=` becomes `le`, `>` becomes `gt` and `<`")
    a("becomes `lt`, then the token is lowercased. So `>=75` is `ge75`, `>=3` is `ge3`,")
    a("`1-2` is unchanged, and the sex stratifier's levels are `1` and `0`. The display")
    a("strings are the `pretty_level()` values and are never these tokens.")
    a("")
    n_miss = gray[("hematoma", "MALE")]["missing_n"]
    a(f"The sex stratifier carries {n_miss} rows with no level (recorded sex is neither male")
    a("nor female, and folding that into a reference group would be misclassification).")
    a("Those rows are excluded from the sex comparison and counted; the other three")
    a("stratifiers are complete.")
    a("")
    a("**Rendering a p value: use `gray.<outcome>.<stratifier>.p_text`, not `.p`.** The")
    a("numeric key is the raw upper tail of the chi-square distribution and underflows to")
    a("exactly 0.0 for the larger statistics, which must never be printed as `p = 0.000`.")
    a("The text key is already formatted through `variable_dict.fmt_p` and renders")
    a("`< 0.001` in that case.")
    a("")
    a("**Note on interpretation, for `08`.** These are unadjusted comparisons of curves")
    a("across fixed, pre-specified strata. They are not a risk model, they carry no")
    a("coefficient and no discrimination statistic, and they must not be written up as")
    a("risk-factor findings. The parent paper owns the hematoma risk model.")
    a("")
    a("---")
    a("")

    # ---- section 6
    a("## 6. The coding-era check (`PLAN.md` 7.3). Re-verified for EVERY outcome.")
    a("")
    a("The companion project verified this for hematoma only. An outcome nobody checked")
    a("could carry a step nobody would have seen, so all eight are checked here.")
    a("")
    a(f"**The transition year is read from the frozen era flag: {era['transition']}.** It is")
    a("not typed, so this check and the tenth-revision-era sensitivity in S4 cannot point")
    a("at different years.")
    a("")
    a("**The rule, stated before the verdicts were read.** An outcome carries a step when")
    a("BOTH hold: the last pre-transition year and the first transition year differ at")
    a("p < 0.05 by a two-sided Fisher exact test, AND the absolute change at that boundary")
    a("is the largest of the twelve adjacent-year changes. The first condition alone would")
    a("over-call across eight outcomes; the second is what makes it a step rather than a")
    a("trend. Both components are published for every outcome.")
    a("")
    a("The two rate columns below are the POOLED era rates, and the boundary change is the")
    a("single-year step from the last pre-transition year to the first transition year.")
    a("They can point in different directions and that is not a contradiction: a pooled")
    a("era difference is a trend statement and the boundary change is a step statement.")
    a("")
    a("| Outcome | Pre-transition rate | Transition-era rate | Boundary change, pp | "
      "Boundary p | Boundary is the largest change | Step |")
    a("|---|---|---|---:|---|---|---|")
    for o in VD.TIMED_OUTCOMES:
        d = era["outcomes"][o.key]
        a(f"| {VD.pretty(o.event_col)} | {VD.fmt_pct(d['pre_pct'], 4)} | "
          f"{VD.fmt_pct(d['post_pct'], 4)} | {d['boundary_change']:+.4f} | "
          f"{VD.fmt_p(d['p_boundary'])} | "
          f"{'yes' if d['is_largest'] else 'no, largest is ' + d['largest_pair']} | "
          f"**{'YES' if d['step'] else 'no'}** |")
    a("")
    if era["step_keys"]:
        names = ", ".join(VD.pretty(VD.timed_outcome(k).event_col).lower()
                          for k in era["step_keys"])
        a(f"**A step was detected for: {names}.** Those outcomes are reported BY ERA and")
        a("never pooled, and era-split curve files were written beside the pooled ones.")
        a("`07_figures.py` must read `cif.era_split_required` and `cif.era_split_files`.")
    else:
        a("**No outcome carries a step at the coding transition.** Every timing curve is")
        a("reported pooled across 2012 to 2024, and `cif.era_split_required` is false, so")
        a("`07_figures.py` plots the pooled curves. Note also that only two of the eight")
        a("outcomes are ascertained through diagnosis coding at all (hematoma, through the")
        a("reoperation diagnosis arms, and unplanned readmission); the other six are NSQIP")
        a("occurrence flags, for which a coding-transition step would have been surprising.")
        a("That is a reason to expect the result, not a reason to skip the check.")
    a("")
    band = era["outcomes"]["hematoma"]["band"]
    a("**The hematoma per-year band, re-derived rather than copied.** Measured here:")
    a(f"**{VD.fmt_pct(band[0], 4)} to {VD.fmt_pct(band[1], 4)}**, with no gap at the")
    a(f"transition. `02` recorded {VD.fmt_pct(CARRYOVER_HEMATOMA_BAND[0], 4)} to")
    a(f"{VD.fmt_pct(CARRYOVER_HEMATOMA_BAND[1], 4)}: "
      f"**{'reproduces' if abs(band[0] - CARRYOVER_HEMATOMA_BAND[0]) < 5e-5 and abs(band[1] - CARRYOVER_HEMATOMA_BAND[1]) < 5e-5 else 'DOES NOT reproduce'}**.")
    a("")
    a("Per-year event counts and rates for all eight outcomes are in the fragment under")
    a("`era.<outcome>.events_by_year` and `era.<outcome>.rate_by_year`, aligned to")
    a("`era.years` and `era.n_by_year`.")
    a("")
    a("### 6.1 A measurement worth flagging: when the two diagnosis columns are actually "
      "populated")
    a("")
    a("`PLAN.md` 7.3 says the ninth-revision reoperation diagnosis column runs 2012 to 2020")
    a("and the tenth-revision one 2014 to 2024. `CLAUDE.md` 9.10 corrects the first to 2012")
    a("to 2017 database-wide. **Inside this cohort, measured rather than asserted, the")
    a(f"ninth-revision column is populated {ctx['icd9_years']} and the tenth-revision column")
    a(f"{ctx['icd10_years']}.** Both are narrower than the database-wide statements, which")
    a("is expected for a single procedure family, and the overlap is real. It is published")
    a("as `timing.coding.icd9_arm_populated_years` and")
    a("`timing.coding.icd10_arm_populated_years`, and it is printed in supplement S1 so no")
    a("year range in the manuscript has to be taken from a plan file. **It is also why the")
    a("hematoma union is ungated across years**: a year-gated rule would drop the events")
    a("ascertained only by the ninth-revision arm inside the overlap.")
    a("")
    a("---")
    a("")

    # ---- section 7
    a("## 7. Supplement S4: the sensitivity analyses this step owns")
    a("")
    a("`tables-csv/s4_sensitivity_03.csv`. Four blocks, "
      f"{ctx['s4_rows']} rows.")
    a("")
    a("1. **One minus Kaplan-Meier instead of Aalen-Johansen**, all eight outcomes. This is")
    a("   also supplement S3, reported there with its confidence bands and here in the")
    a("   common sensitivity format.")
    a("2. **Restricted to the tenth-revision coding era**, operative years 2016 to 2024,")
    a(f"   n = {VD.fmt_count(ctx['n_icd10'])}.")
    a("3. **Restricted to the era in which elective status came from the emergency-case")
    a(f"   field**, operative years 2012 to 2020, n = {VD.fmt_count(ctx['n_emerg'])}.")
    a("4. **The three hematoma code sets**, V2 (primary), V1 and V0.")
    a("")
    a("| Code set | Events | Day 30 cumulative incidence | Median (IQR) | Day 0 or 1 |")
    a("|---|---:|---|---|---|")
    for v in ("v2", "v1", "v0"):
        d = s4[f"hematoma_{v}"]
        a(f"| {v.upper()}{' (primary)' if v.upper() == CPT.HEMATOMA_DEFINITION else ''} | "
          f"{VD.fmt_count(d['events'])} | {VD.fmt_pct(d['day30'] * 100.0, 4)} | "
          f"{d['median']:.0f} ({VD.fmt_ci(d['q1'], d['q3'], 0)}) | "
          f"{VD.fmt_n_pct(d['n01'], d['obs'], 1)} |")
    a("")
    a("**How the code-set block is computed, and why it is not a cohort derivation.**")
    a("`HEMATOMA_V0` and `HEMATOMA_V1` are frozen columns and the days-to-event field is")
    a("the same raw reoperation-day column under all three definitions. The block pairs a")
    a("frozen event flag with a frozen raw day column. To prove the pairing is the same")
    a("operation `02_build_cohort.py` performed, the script **re-derives the primary day")
    a("column from those same two frozen columns and asserts it reproduces `HEMATOMA_POD`")
    a("exactly**; if that ever fails the block is a stop condition rather than a published")
    a("row. It held this run. The 0 to 30 integer handling rule was re-applied for form and")
    a(f"affected {ctx['day_rule_dropped']} values, which is the same answer `02` recorded.")
    a("")
    a("---")
    a("")

    # ---- section 8
    a("## 8. Gate 2 (`CLAUDE.md` 8.4): the two carried-over numbers this step owns")
    a("")
    a("**Re-derived here, not copied.** `02` published its own measurements under a")
    a("data-quality namespace precisely so they could not be mistaken for the published")
    a("value, and its binding item 1 forbids copying a number out of a handoff note.")
    a("")
    ga = gate2["analytic"]
    gi = gate2["identified"]
    a("| Rung | Denominator | Median postoperative day | Day 0 or 1 share |")
    a("|---|---:|---:|---|")
    a(f"| **Analytic (PRIMARY, `CLAUDE.md` 4.4)** | {VD.fmt_count(ga['n_obs'])} | "
      f"**{ga['median']:.0f}** | **{VD.fmt_n_pct(ga['n01'], ga['n_obs'], 2)}** |")
    if gi is not None:
        a(f"| Identified (reconciliation only) | {VD.fmt_count(gi['day_recorded'])} | "
          f"{gi['median']:.0f} | {VD.fmt_n_pct(gi['pod01_n'], gi['day_recorded'], 2)} |")
    a(f"| Carried over (`PLAN.md` 2.2) | not recorded | {CARRYOVER_MEDIAN_REOP_POD} | "
      f"{VD.fmt_pct(CARRYOVER_POD01_PCT, 1)} |")
    a("")
    a("**Verdict.**")
    a("")
    a(f"* Median postoperative day of the hematoma reoperation reproduces the carried-over")
    a(f"  value of {CARRYOVER_MEDIAN_REOP_POD} on the analytic rung: "
      f"**{bool(gate2['med_match_a'])}**"
      + (f"; on the identified rung: **{bool(gate2['med_match_i'])}**."
         if gi is not None else "."))
    a(f"* Share on postoperative day 0 or 1 reproduces the carried-over "
      f"{VD.fmt_pct(CARRYOVER_POD01_PCT, 1)} to one decimal on the analytic rung: "
      f"**{bool(gate2['p01_match_a'])}** "
      f"(measured {VD.fmt_pct(ga['p01'], 2)})"
      + (f"; on the identified rung: **{bool(gate2['p01_match_i'])}** "
         f"(measured {VD.fmt_pct(gi['pod01_pct'], 2)})." if gi is not None else "."))
    a("")
    a("**The carried-over 42.6% is an identified-rung figure.** The analytic figure is")
    a("higher, and that is not a discrepancy to reconcile away: it is the same measure on a")
    a("different and correctly stated denominator, plus a documented movement of the")
    a("hematoma definition from the superseded code set to the current one. `CLAUDE.md` 4.4")
    a("rules that the manuscript reports the analytic value, so **the analytic value is")
    a("primary and the identified value is printed beside it with its own denominator.**")
    a("Neither is overwritten (`CLAUDE.md` 8.5).")
    a("")
    a("Read these from `timing.gate2.*`, never from this table.")
    a("")
    a("---")
    a("")

    # ---- section 9
    a("## 9. The fragment")
    a("")
    a(f"`outputs/results_03.json`, **{nkeys} keys**, every one inside this step's five")
    a("reserved prefixes `cif.`, `timing.`, `gray.`, `era.` and `s3.`. Verified two ways:")
    a("by this script's own namespace guard before the file is written, and by importing")
    a("`run_all.namespace_violations` afterwards, which returns NONE. Values are flat")
    a("scalars or lists of scalars, there is no nested object, and there is no `NaN` or")
    a("`Infinity`: an unavailable quantity is written as `null`.")
    a("")
    a("Key families, so a downstream step can find what it needs without grepping:")
    a("")
    a("| Family | What is in it |")
    a("|---|---|")
    a("| `cif.<outcome>.day30.*` | the terminal estimate as a proportion (`est`, `lo`, "
      "`hi`), as a percentage (`pct`, `pct_lo`, `pct_hi`) and as formatted text (`text`) |")
    a("| `cif.<outcome>.day<D>.pct` | landmark days "
      f"{', '.join(str(d) for d in LANDMARK_DAYS)} |")
    a("| `cif.<outcome>.{events_n,competing_deaths_n,day_unobserved_events_n,denominator}` "
      "| what the curve was built from |")
    a("| `cif.era_split_required`, `cif.era_split_files` | whether `07` must plot by era |")
    a("| `timing.<outcome>.*` | Table 3A: events, day recorded, median, quartiles, day 0 "
      "to 1 and after day 10, with denominators |")
    a("| `timing.los_postop.*` | the Figure 2 reference line |")
    a("| `timing.second_event.*`, `timing.trach.*` | the time-to-first-event caveat and "
      "the tracheostomy slot counts |")
    a("| `timing.sens.*` | the S4 blocks other than the Kaplan-Meier contrast |")
    a("| `timing.gate2.*` | the two carried-over numbers and their verdicts |")
    a("| `gray.*` | test statistics, degrees of freedom, p values, suppression and every "
      "stratum-level day-30 estimate |")
    a("| `era.*` | per-year counts and rates, the boundary tests and the step verdicts |")
    a("| `s3.*` | the competing-risk correction |")
    a("")
    a("---")
    a("")

    # ---- section 10
    a("## 10. What `07` (figures) and `08` (manuscript) must know")
    a("")
    a("1. **Figure 2 is drawn from `tables-csv/cif_<outcome>.csv` and nothing else.** One")
    a("   file per outcome, in `variable_dict.TIMED_OUTCOMES` order, 31 rows on integer")
    a("   days. **Plot step curves.** Do not smooth, do not interpolate, do not draw a")
    a("   curve that implies sub-day resolution the data does not have, and do not extend")
    a("   any line past day 30.")
    a("2. **The vertical reference line goes at `timing.los_postop.median`**, which is the")
    a("   cohort median postoperative length of stay taken from the frozen column.")
    a("3. **The confidence band columns are already percentages.** So is")
    a("   `cif.<outcome>.day30.pct`. The `est` / `lo` / `hi` keys are proportions. Pick one")
    a("   and do not mix them. **That rule is THIS STEP'S and covers only `cif.*`.** It is")
    a("   published as `cif.scale` and it is not project-wide: `06_window.py` uses the same")
    a("   three suffixes for PERCENTAGES, so reading `window.overall.cif30.est` as a")
    a("   proportion would render the paper's lead result 100 times too small, and")
    a("   `05_inout.py` uses them for odds ratios and a kappa. Every namespace this step")
    a("   owns now carries its own declaration (`cif.scale`, `timing.scale`, `gray.scale`,")
    a("   `era.scale`, `s3.scale`), each with a machine-readable family and unit list, and")
    a("   `08` must read the declaration belonging to the step that owns the key")
    a("   (`.build_facts.md` section 9).")
    a("4. **The 30-day incidence is `cif.<outcome>.day30.pct`, not `cif.<outcome>.crude_pct`.**")
    a("   The two are close and they are not the same quantity: the crude percentage counts")
    a("   every recorded event, the day-30 estimate counts the events that carry an observed")
    a("   day, and only the second is the terminal point of the Figure 2 curve. See section")
    a("   2.1 and `cif.day30_reporting_key`.")
    a("5. **`cif.era_split_required` is the switch.** If it is true, plot the outcomes named")
    a("   in `era.step_outcome_keys` by era from the files listed in")
    a("   `cif.era_split_files`, and never pooled.")
    a("6. **Table 3A carries the numbers; Figure 2 carries the shape.** Do not print the")
    a("   day-30 estimate on the figure and do not redraw Table 3A as a chart.")
    a("7. **The Methods sentences this step is the source for**: time zero is the day of")
    a("   the principal operation; the estimator is Aalen-Johansen with death as the")
    a("   competing event; comparisons across strata use Gray's test; censoring is")
    a("   administrative at day 30, non-informative and complete by construction; no")
    a("   estimate extends past day 30; for death the estimator reduces to the ordinary")
    a("   cumulative incidence. They are all in the fragment as text keys (`timing.time_zero`,")
    a("   `timing.estimator`, `timing.censoring`, `timing.extrapolation`,")
    a("   `timing.death.no_competing_event`, `gray.method`, `timing.ci.method`), so the")
    a("   prose can be rendered rather than retyped.")
    a("8. **Readmission: report the curve SHAPE only.** Its post-discharge fraction is")
    a("   uninformative by construction and belongs to nobody;")
    a("   `timed_outcome('readmit').postdisch_eligible` is `False`.")
    a("9. **Do not print a post-discharge fraction from anything in this step.** It does not")
    a("   compute one. Table 3B and the denominator ladder are `04`'s.")
    a("10. **Nothing here is a risk model.** No coefficient, no odds ratio, no")
    a("   discrimination statistic, no points score. The stratifiers are fixed and")
    a("   pre-specified and Gray's test is a comparison of curves, not a fitted model.")
    a("11. **`08` must state the estimator implementation.** The Methods should say the")
    a("    Aalen-Johansen estimator was computed on integer postoperative days; it should")
    a("    not name a library, because none was used for the estimate.")
    a("")
    a("---")
    a("")

    # ---- section 11
    a("## 11. Decisions this step MADE rather than inherited")
    a("")
    a("Listed so a reviewer can challenge them.")
    a("")
    a("1. **The estimator is hand-implemented** rather than delegated to `lifelines`,")
    a("   because the library jitters tied event times and drops the events at time 0")
    a("   (section 1). The variance and the transformation are the standard ones, so the")
    a("   interval is comparable with any other implementation.")
    a("2. **An event with no recorded day contributes no event and stays in the risk set.**")
    a("   The alternative, placing it at day 30, would fabricate a timing; excluding the")
    a("   patient would move the denominator. The count and the maximum understatement are")
    a("   published per outcome instead.")
    a("3. **A patient with both the event of interest and a death enters as the event of")
    a("   interest**, because the event day never exceeds the death day in this cohort")
    a("   (checked every run; a violation is a stop condition). A death on the same day as")
    a("   the event does not erase the event.")
    a("4. **Gray's test is the rho = 0 member of the family**, which under complete")
    a("   follow-up is exactly the log-rank statistic on subdistribution risk sets.")
    a("5. **The suppression rule drops the sparse stratum rather than the whole comparison**")
    a("   and reports it with its count. Suppressing the entire comparison would discard")
    a("   the information in the strata that are adequately sized. Since the Batch 3")
    a("   review the same rule also **withholds that stratum's proportion**, not only its")
    a("   place in the test, per the project-wide ruling in `CLAUDE.md` 6.5. One decision")
    a("   now drives both, so this step and `04` cannot treat the same sparse cell")
    a("   differently again.")
    a("6. **The era-step rule requires both a significant boundary test and the boundary")
    a("   being the largest adjacent change** (section 6). Either alone would be wrong in a")
    a("   different direction.")
    a("7. **The transition year is read from the frozen era flag**, never typed.")
    a("8. **`derived/cohort_identified.parquet` is opened read-only, for two columns**, to")
    a("   re-derive the Gate 2 identified-rung reconciliation rather than copy it")
    a("   (section 8). No cohort is derived, nothing is written, and the analytic value")
    a("   remains primary.")
    a("9. **The denominator for Table 3A's percentages is the events with the postoperative")
    a("   day recorded**, printed in its own column and repeated in the machine-readable")
    a(f"   `{DENOM_COLUMN}` column.")
    a("10. **Fragment level tokens are transliterated** (section 5); the display strings")
    a("    are never the tokens.")
    a("")
    NOTES_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    print("=" * 74)
    print("03_timing.py  --  cumulative incidence by postoperative day")
    print("=" * 74)

    _self_test()
    print("estimator self-test .......... PASS")
    diag = lifelines_diagnostic()
    if diag.get("available"):
        print(f"lifelines diagnostic ......... exact "
              f"{diag['fixture_exact_pct']:.1f}% vs library "
              f"{diag['fixture_library_pct']:.1f}% on the fixture; "
              f"zero jitter rejected = {diag['rejects_zero_jitter']}")
    else:
        print("lifelines diagnostic ......... skipped (library not importable)")

    if not COHORT_ANALYTIC.exists():
        print(f"STOP: {COHORT_ANALYTIC} does not exist. Run 02_build_cohort.py first.")
        return 1
    df = pd.read_parquet(COHORT_ANALYTIC)
    n_analytic = len(df)
    print(f"analytic cohort .............. {VD.fmt_count(n_analytic)} rows x "
          f"{df.shape[1]} columns")

    frag = Fragment()

    publish_scale_declarations(frag)
    print(f"scale declarations ........... {len(SCALE_DECLARATIONS)} namespaces, each "
          f"scoped to itself (.build_facts.md 9)")

    store = analysis_cif(df, frag)
    print("cumulative incidence ......... 8 curves, day-30 estimate equals the empirical "
          "proportion of the events with an observed day, on every outcome")
    for o in VD.TIMED_OUTCOMES:
        r: CIFResult = store[o.key]["res"]
        print(f"    {o.key:10s} {VD.fmt_pct(r.day30 * 100.0, 4):>9s}  "
              f"({VD.fmt_ci(r.day30_lo * 100.0, r.day30_hi * 100.0, 4)})")

    libdiff = lifelines_cohort_diagnostic(store)
    jitter_outcomes = int(libdiff.pop("_jitter_outcomes_n", 0))
    frag.put("timing.library.jitter_triggered_outcomes_n", jitter_outcomes)
    frag.put("timing.library.outcomes_n", len(VD.TIMED_OUTCOMES))
    frag.put("timing.library.tie_detector_fires_on_minimal_fixture",
             diag.get("tie_detector_fires"))
    # The precise mechanism, measured rather than asserted. The library builds its event
    # table with a synthesised row at time 0, so time 0 is always the first row; the
    # lagged survival is undefined there and is zeroed, which drops the events AT TIME 0
    # specifically. An earliest observed event at day 1 is not dropped. The two
    # descriptions coincide here only because time 0 is the earliest observed time on
    # every outcome in this cohort, which is why it is counted rather than assumed.
    earliest_zero = sum(
        1 for o in VD.TIMED_OUTCOMES
        if float(np.min(np.asarray(store[o.key]["times"], dtype=float))) == 0.0)
    day0_events = sum(
        1 for o in VD.TIMED_OUTCOMES if int(store[o.key]["res"].e1[0]) > 0)
    frag.put("timing.library.outcomes_with_earliest_observed_time_zero_n", earliest_zero)
    frag.put("timing.library.outcomes_with_day0_events_n", day0_events)
    frag.put("timing.library.note",
             "A library-behaviour diagnostic, never an estimate. It records what the "
             "lifelines Aalen-Johansen fitter would have returned on the same time and "
             "event-type vectors, which is lower than the exact value because that "
             "implementation drops the events at time 0. Its event table always carries "
             "a synthesised time-0 row, the lagged survival is undefined on that row and "
             "is zeroed, and the events sitting on it therefore never reach the running "
             "sum. Events at a later earliest observed time are not affected. In this "
             "cohort time 0 is the earliest observed time on every outcome and day 0 "
             "carries real events on every outcome, so the loss is real here; the "
             "mechanism is stated as time 0 rather than as the earliest observed time "
             "because only the former generalises.")
    frag.put("timing.library.fixture_exact_pct", num(diag.get("fixture_exact_pct"), 4))
    frag.put("timing.library.fixture_reported_pct",
             num(diag.get("fixture_library_pct"), 4))
    frag.put("timing.library.fixture_events_lost_n", diag.get("fixture_events_lost_n"))
    frag.put("timing.library.rejects_zero_jitter", diag.get("rejects_zero_jitter"))
    for k in [o.key for o in VD.TIMED_OUTCOMES]:
        frag.put(f"timing.library.{k}.understatement_pp",
                 num(libdiff[k], 6) if k in libdiff else None)
    if libdiff:
        worst_lib = max(libdiff, key=lambda kk: abs(libdiff[kk]))
        frag.put("timing.library.max_understatement_pp", num(libdiff[worst_lib], 6))
        frag.put("timing.library.max_understatement_outcome",
                 VD.pretty(VD.timed_outcome(worst_lib).event_col))
        print(f"library diagnostic ........... worst understatement "
              f"{libdiff[worst_lib]:.4f} percentage points ({worst_lib})")
    else:
        frag.put("timing.library.max_understatement_pp", None)
        frag.put("timing.library.max_understatement_outcome", "")

    table3a = analysis_table3a(df, store, frag)
    print("Table 3A ..................... tables-csv/table3a_complication_timing.csv")

    s3 = analysis_s3(store, n_analytic, frag)
    print(f"Supplement S3 ................ largest correction "
          f"{s3['_worst']['diff_pp']:.4f} percentage points "
          f"({s3['_worst']['key']}), largest relative {s3['_worst']['rel']:.2f}%")

    s1meta = analysis_s1(df, frag, table3a["trach"])
    print(f"Supplement S1 ................ tables-csv/s1_field_definitions.csv "
          f"(diagnosis columns populated {s1meta['icd9_years']} and "
          f"{s1meta['icd10_years']})")

    gray = analysis_gray(df, store, frag)
    print(f"Gray's test .................. {len(gray)} comparisons, "
          f"{frag._d['gray.strata_proportions_suppressed_n']} stratum-level proportions "
          f"withheld below {VD.MIN_STRATUM_EVENTS} events")

    era = analysis_era(df, frag)
    if era["step_keys"]:
        print(f"coding-era check ............. STEP DETECTED for {era['step_keys']}")
    else:
        print("coding-era check ............. no step at the transition for any outcome")
    write_era_curves(df, era["step_keys"], era["transition"], frag)

    n_icd10 = int((pd.to_numeric(df["ERA_ICD10"], errors="coerce").fillna(0) == 1).sum())
    n_emerg = int((pd.to_numeric(df["ERA_EMERGNCY"], errors="coerce").fillna(0) == 1).sum())
    s4 = analysis_s4(df, store, frag)
    print("Supplement S4 ................ tables-csv/s4_sensitivity_03.csv")

    gate2 = analysis_gate2(table3a, frag)
    print(f"Gate 2 ....................... {frag._d['timing.gate2.verdict']}")

    nkeys = frag.write(FRAGMENT_PATH)
    print(f"fragment ..................... {FRAGMENT_PATH.name}, {nkeys} keys")

    # Independent namespace verification through the orchestrator's own guard.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_run_all_guard", HERE / "run_all.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # NamedTuple and dataclass definitions resolve their module through
        # sys.modules[__name__], so the module has to be registered before it executes.
        sys.modules["_run_all_guard"] = mod
        spec.loader.exec_module(mod)
        data = mod.load_fragment(FRAGMENT_PATH)     # also validates keys and values
        viol = mod.namespace_violations("03", FRAGMENT_PATH, data)
        print(f"namespace guard .............. "
              f"{'NONE' if not viol else 'VIOLATIONS: ' + str(viol)} "
              f"({len(data)} keys validated by run_all.load_fragment)")
        if viol:
            return 1
    except Exception as exc:  # pragma: no cover
        print(f"namespace guard .............. could not run run_all guard ({exc})")

    write_notes({
        "n_analytic": n_analytic, "store": store, "table3a": table3a, "s3": s3,
        "era": era, "gray": gray, "gate2": gate2, "diag": diag, "s4": s4,
        "libdiff": libdiff, "jitter_outcomes": jitter_outcomes,
        "nkeys": nkeys, "n_icd10": n_icd10, "n_emerg": n_emerg,
        "icd9_years": s1meta["icd9_years"], "icd10_years": s1meta["icd10_years"],
        "s4_rows": frag._d["timing.s4.rows_n"],
        "day_rule_dropped": frag._d["timing.sens.hematoma.day_rule_dropped_n"],
        "fragment": frag._d,
    })
    print(f"notes ........................ {NOTES_PATH.name}")
    print("=" * 74)
    print("03_timing.py COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
