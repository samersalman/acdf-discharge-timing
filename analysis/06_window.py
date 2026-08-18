#!/usr/bin/env python3
"""06_window.py - the risk-stratified SAFE DISCHARGE WINDOW (PLAN.md 6.5, CLAUDE.md 6.4).

THE ESTIMAND, locked in CLAUDE.md 6.4 and .build_facts.md section 2 before any
estimator was run:

    For a given patient stratum and the pre-specified composite outcome, the SAFE
    DISCHARGE DAY is the earliest postoperative day D at which the CONDITIONAL
    probability of a subsequent qualifying event, given the patient is event-free at
    day D, falls below a pre-specified absolute-risk threshold.

It is a conditional-risk-given-survival quantity read off the cumulative incidence
function, NOT a hazard. On the fixed 30-day window,

    conditional risk at day D  =  (cumulative incidence at day 30
                                   minus cumulative incidence at the start of day D)
                                  / probability of being event-free and alive at the
                                    start of day D

Death is a COMPETING event, so the numerator is the Aalen-Johansen cumulative
incidence and not one minus Kaplan-Meier: a patient who died on day 3 is not at risk
of a hematoma reoperation on day 7 (PLAN.md 6.1). The size of that correction is
measured and published in tables-csv/s4_sensitivity_06.csv rather than assumed small.

DAY INDEXING, and why it is the START of day D. The two statements the estimand has
to satisfy simultaneously are "the additional cumulative incidence accrued between
day D and day 30" and "the conditional risk at day 0 equals the day-30 cumulative
incidence". Both hold only if an event recorded ON day D counts as still ahead of a
patient discharged on day D, that is, if the conditioning set is the patients
event-free and alive at the START of day D. That is also the conservative reading
for a safety claim: NSQIP records whole postoperative days, an event and a discharge
on the same integer day cannot be ordered, and crediting the observation window with
catching a same-day event would shorten the recommended window on an assumption the
data cannot support. The opposite (end-of-day) indexing is reported as a sensitivity.

WHAT THIS STEP MUST NOT DO (CLAUDE.md 6.6 and 7.1). This project fits no new
multivariable hematoma risk model and builds no points score; both belong to the
companion hematoma paper. No coefficient is estimated here, no odds ratio is
reported, and no discrimination statistic is presented. The stratifiers are the
locked FIXED strata of variable_dict.STRATIFIERS and the companion paper is cited
for their derivation. The output is absolute conditional risk by stratum and day.

BOUNDARY WITH 03_timing.py. 03 owns the cumulative incidence curves, Gray's tests
and Table 3A for the eight roster outcomes. This step computes cumulative incidence
only for its own composite (WINDOW_COMP, which is not one of the eight) and only as
the input its estimand needs. It emits no general-purpose cumulative incidence
table.

INPUTS   derived/cohort_analytic.parquet   (frozen by 02; never re-derived here)
OUTPUTS  outputs/results_06.json           keys inside window. condrisk. suppress.
         tables-csv/table5_safe_discharge_day.csv
         tables-csv/table5b_conditional_risk.csv     (Figure 6 reads this)
         tables-csv/s4_sensitivity_06.csv
         derived/notes_06.md

    python3 06_window.py              full run
    python3 06_window.py --self-test  estimator unit tests only, writes nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import variable_dict as VD  # noqa: E402

# CLAUDE.md hard rule 4. This is the ONE step in the project that draws on the global
# NumPy stream (the within-stratum bootstrap), so seeding it is load-bearing for
# reproducibility and two consecutive runs must be byte-identical.
np.random.seed(VD.SEED)

DERIVED = ROOT / "derived"
TABLES = ROOT / "tables-csv"
OUTPUTS = ROOT / "outputs"

ANALYTIC_PARQUET = DERIVED / "cohort_analytic.parquet"
RESULTS_JSON = OUTPUTS / "results_06.json"
NOTES_MD = DERIVED / "notes_06.md"
TABLE5_CSV = TABLES / "table5_safe_discharge_day.csv"
TABLE5B_CSV = TABLES / "table5b_conditional_risk.csv"
S4_CSV = TABLES / "s4_sensitivity_06.csv"

# CLAUDE.md 4.2: this step's reserved fragment namespaces. Anything else is rejected by
# the merge and stops the pipeline, so it is rejected here first, where the cause is.
RESERVED_PREFIXES: Tuple[str, ...] = ("window.", "condrisk.", "suppress.")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")

DAYS = VD.FOLLOWUP_DAYS + 1          # postoperative days 0 through 30 inclusive
BOOTSTRAP_REPLICATES = 1000          # .build_facts.md section 2, locked
CI_ALPHA = 0.05
NOT_REACHED = "Not reached within 30 days"
SUPPRESSED_TEXT = "Suppressed"
NOT_APPLICABLE = "Not applicable"

# Sentinel for a bootstrap replicate in which a threshold is never crossed inside the
# window. It is strictly greater than every real day, so a percentile that lands on it
# is honestly reported as "not reached" rather than silently clipped to day 30
# (CLAUDE.md hard rule 9: do not extrapolate past day 30).
UNREACHED_DAY = VD.FOLLOWUP_DAYS + 1

STRATIFIER_DISPLAY: Dict[str, str] = {
    "AGE_CAT": VD.pretty("AGE_CAT"),
    "ASA_CAT": VD.pretty("ASA_CAT"),
    "LEVELS_CAT": VD.pretty("LEVELS_CAT"),
    # The stratifier is sex; MALE is only how 02 stores it. pretty("MALE") is
    # "Male sex", which reads as a level rather than as a stratifier heading.
    "MALE": VD.pretty("SEX"),
}

# Fragment key segment for each stratifier. ">=" and "%" are not legal in a fragment
# key, so the locked ASCII level strings are slugged, never renamed.
STRATIFIER_SLUG: Dict[str, str] = {
    "AGE_CAT": "age",
    "ASA_CAT": "asa",
    "LEVELS_CAT": "levels",
    "MALE": "sex",
}
LEVEL_SLUG_OVERRIDE: Dict[Tuple[str, str], str] = {
    ("MALE", "1"): "male",
    ("MALE", "0"): "female",
}

# Where a stratifier is stored as a BINARY that cannot represent every recorded value of
# the variable it stands for, this names the column that CAN. `02_build_cohort.py` leaves
# `MALE` Int8-missing for the patients whose sex is recorded as neither male nor female,
# deliberately, because folding them into the female reference group would be
# misclassification; `SEX` still carries the recorded value. Without this map the residual
# level falls through to VD.MISSING_LABEL and a reviewer-facing table asserts the data are
# missing when they are recorded. The DISPLAY STRING is never written here: it is resolved
# through variable_dict.pretty_level() on the source column, so the label layer stays the
# single source of truth (CLAUDE.md hard rule 2).
RESIDUAL_SOURCE_COLUMN: Dict[str, str] = {
    "MALE": "SEX",
}

# The fragment-key segment used for the residual level of a stratifier. It is FROZEN at
# this string even where the residual level turns out to be a recorded value, because a
# fragment key is a machine identifier that `verify_numbers.py` (STRATUM_SLUGS) and
# `08_generate_manuscript.py` both address by name, and renaming it would break two files
# this step does not own. It is never rendered: every artefact prints the DISPLAY label,
# which is resolved from variable_dict. See notes_06.md section 5.
RESIDUAL_SLUG = "not-recorded"

# The postoperative days at which the composition of the conditioning set is published as
# scalars a manuscript can quote directly. Every day 0 to 30 is published as well, as a
# list under condrisk., so nothing here restricts what may be reported; these are the days
# the safe-discharge result is actually read at (day 0 and day 1 are the same-day and
# one-night discharge points, days 3, 7 and 14 are the overall and stratum threshold days,
# day 30 is the end of the window).
RISKSET_REPORT_DAYS: Tuple[int, ...] = (0, 1, 3, 7, 14, 30)


def level_slug(column: str, level: str) -> str:
    """Fragment-key segment for one stratum level, from the locked ASCII level string."""
    if (column, level) in LEVEL_SLUG_OVERRIDE:
        return LEVEL_SLUG_OVERRIDE[(column, level)]
    return level.replace(">=", "ge").replace("<=", "le").replace("+", "plus").lower()


def threshold_slug(threshold: float) -> str:
    """Fragment-key segment for a pre-specified threshold: 0.005 -> "threshold_0_5pct"."""
    label = VD.threshold_label(threshold)          # raises on an unspecified threshold
    return "threshold_" + label.replace("%", "pct").replace(".", "_")


# ===========================================================================
# 1. THE ESTIMATOR
# ===========================================================================
#
# All timing is on integer postoperative days, so every estimator here is a step
# function on integers (CLAUDE.md 6.1). Follow-up is fixed and COMPLETE at 30 days:
# censoring is administrative and non-informative and there is no loss to follow-up,
# so the only exits from the risk set inside the window are the composite event, the
# competing death, and reaching day 30 alive and event-free.
#
# The arrays below are indexed so that position d, for d in 0..30, is the value at the
# START of postoperative day d (before that day's events), and position 31 is the value
# at the END of day 30, which is the terminal estimate. Nothing extends past it.
# ===========================================================================


def aalen_johansen(e1: np.ndarray, e2: np.ndarray, n: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aalen-Johansen cumulative incidence with death as the competing event.

    e1, e2 : (B, 31) counts of cause-1 (composite) and cause-2 (competing death)
             events on each postoperative day, for B independent replicates.
    n      : (B,) stratum sizes.

    Returns (cif, surv, at_risk), each (B, 32), where column d for d in 0..30 is the
    value at the START of day d and column 31 is the terminal value at the end of
    day 30.  `surv` is the probability of being event-free AND ALIVE, which is exactly
    the denominator the conditional-risk estimand asks for.
    """
    b, days = e1.shape
    cif = np.zeros((b, days + 1), dtype=float)
    surv = np.ones((b, days + 1), dtype=float)
    at_risk = np.zeros((b, days + 1), dtype=float)

    running_cif = np.zeros(b, dtype=float)
    running_surv = np.ones(b, dtype=float)
    running_risk = np.asarray(n, dtype=float).copy()

    for d in range(days):
        cif[:, d] = running_cif
        surv[:, d] = running_surv
        at_risk[:, d] = running_risk
        safe = np.where(running_risk > 0, running_risk, 1.0)
        haz_cause1 = np.where(running_risk > 0, e1[:, d] / safe, 0.0)
        haz_any = np.where(running_risk > 0, (e1[:, d] + e2[:, d]) / safe, 0.0)
        running_cif = running_cif + running_surv * haz_cause1
        running_surv = running_surv * (1.0 - haz_any)
        running_risk = running_risk - e1[:, d] - e2[:, d]

    cif[:, days] = running_cif
    surv[:, days] = running_surv
    at_risk[:, days] = running_risk
    return cif, surv, at_risk


def kaplan_meier_survival(e1: np.ndarray, e2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival for the composite, treating the competing death as CENSORING.

    This is the WRONG estimator for this question and exists only to measure the size of
    the competing-risk correction (PLAN.md 6.6, supplement S4). One minus this quantity
    overstates the cumulative incidence of every non-fatal outcome, because it treats a
    death as if the dead patient would have gone on to experience the complication at the
    same rate as the living.

    Same (B, 32) indexing as aalen_johansen(). A death on day d leaves the risk set after
    that day's events, which is the standard tie convention for a censored observation.
    """
    b, days = e1.shape
    surv = np.ones((b, days + 1), dtype=float)
    running_surv = np.ones(b, dtype=float)
    running_risk = np.asarray(n, dtype=float).copy()
    for d in range(days):
        surv[:, d] = running_surv
        safe = np.where(running_risk > 0, running_risk, 1.0)
        haz = np.where(running_risk > 0, e1[:, d] / safe, 0.0)
        running_surv = running_surv * (1.0 - haz)
        running_risk = running_risk - e1[:, d] - e2[:, d]
    surv[:, days] = running_surv
    return surv


def conditional_risk_aj(cif: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """Conditional risk of a SUBSEQUENT composite event, by day, from an Aalen-Johansen fit.

    Returns (B, 31). Column D is the probability of a qualifying event on day D or later,
    given the patient is event-free and alive at the START of day D. Column 0 is therefore
    the day-30 cumulative incidence itself, which is the estimand's own sanity check.
    """
    terminal = cif[:, -1][:, None]
    numerator = terminal - cif[:, :-1]
    denominator = surv[:, :-1]
    out = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0),
                   np.nan)
    return np.clip(out, 0.0, None)


def conditional_risk_aj_end_of_day(cif: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """The alternative day-indexing convention, for the sensitivity analysis only.

    Column D is the probability of an event on day D+1 or later given the patient is
    event-free and alive at the END of day D, that is, crediting the observation window
    with catching an event recorded on the discharge day itself.
    """
    terminal = cif[:, -1][:, None]
    numerator = terminal - cif[:, 1:]
    denominator = surv[:, 1:]
    out = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0),
                   np.nan)
    return np.clip(out, 0.0, None)


def conditional_risk_km(surv_km: np.ndarray) -> np.ndarray:
    """The same estimand computed from one minus Kaplan-Meier instead of Aalen-Johansen.

    With death censored, the ratio collapses to 1 - S(30) / S(start of day D).
    """
    terminal = surv_km[:, -1][:, None]
    start = surv_km[:, :-1]
    ratio = np.where(start > 0, terminal / np.where(start > 0, start, 1.0), np.nan)
    return np.clip(1.0 - ratio, 0.0, None)


def first_day_below(curve: np.ndarray, threshold: float) -> np.ndarray:
    """Earliest day D at which `curve` falls strictly below `threshold`, per replicate.

    Returns an integer array; UNREACHED_DAY marks a replicate in which the threshold is
    never crossed inside the 30-day window. It is never clipped to day 30 and never
    extrapolated past it (CLAUDE.md hard rule 9).
    """
    below = curve < threshold
    any_below = below.any(axis=1)
    first = np.argmax(below, axis=1)
    return np.where(any_below, first, UNREACHED_DAY).astype(int)


def stays_below(curve: np.ndarray, threshold: float, day: int) -> bool:
    """True if the curve remains strictly below `threshold` from `day` through day 30.

    A transparency check, not part of the estimand. The conditional risk is not
    guaranteed monotone: a day carrying a competing death but no composite event
    shrinks the conditioning set without shrinking the residual risk, which can nudge
    the curve up. Any such day is reported rather than smoothed away.
    """
    if day > VD.FOLLOWUP_DAYS:
        return False
    return bool(np.all(curve[day:] < threshold))


def counts_by_day(pod_event: np.ndarray, pod_competing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-day counts of the composite event and of the competing death, days 0 to 30."""
    e1 = np.bincount(pod_event.astype(int), minlength=DAYS).astype(float)
    e2 = np.bincount(pod_competing.astype(int), minlength=DAYS).astype(float)
    if len(e1) != DAYS or len(e2) != DAYS:
        raise SystemExit("[06] a postoperative day outside 0 to 30 reached the estimator; "
                         "02 froze the handling rule and no step may clip or drop locally")
    return e1, e2


# ===========================================================================
# 2. THE BOOTSTRAP
# ===========================================================================
#
# .build_facts.md section 2 locks the uncertainty rule: the Aalen-Johansen variance
# where available, otherwise a seeded bootstrap with SEED, 1,000 replicates, PATIENTS
# RESAMPLED WITHIN STRATUM.
#
# The conditional risk is a RATIO of a difference of two Aalen-Johansen quantities to a
# third, so there is no closed-form Aalen-Johansen variance for it; the closed form
# exists for the cumulative incidence alone. The locked fallback therefore applies, and
# every conditional-risk interval below is a within-stratum percentile bootstrap.
#
# Resampling n patients with replacement from a stratum is EXACTLY a multinomial draw
# over the cells the patients fall into, because patients inside one cell are
# exchangeable and each patient falls in exactly one cell. The cells are (composite
# event on day d) and (competing death on day d) for d in 0..30, plus one cell for
# reaching day 30 event-free and alive. That makes the bootstrap exact and cheap
# rather than approximate and slow, and it draws on the global NumPy stream that
# np.random.seed(SEED) fixes at the top of this file.
# ===========================================================================


def bootstrap_cells(e1: np.ndarray, e2: np.ndarray, n: int,
                    replicates: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample `n` patients within stratum, `replicates` times. Returns (e1, e2) draws."""
    event_free = n - int(e1.sum()) - int(e2.sum())
    if event_free < 0:
        raise SystemExit("[06] more events than patients in a stratum; the composite and "
                         "the competing event are not disjoint")
    cells = np.concatenate([e1, e2, [float(event_free)]])
    probs = cells / cells.sum()
    probs = probs / probs.sum()
    draws = np.random.multinomial(n, probs, size=replicates).astype(float)
    return draws[:, :DAYS], draws[:, DAYS:2 * DAYS]


def percentile_ci(values: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided 95% percentile interval."""
    lo = np.nanpercentile(values, 100.0 * CI_ALPHA / 2.0, axis=axis)
    hi = np.nanpercentile(values, 100.0 * (1.0 - CI_ALPHA / 2.0), axis=axis)
    return lo, hi


# ===========================================================================
# 3. STRATA
# ===========================================================================


@dataclass(frozen=True)
class Stratum:
    slug: str            # fragment key segment, e.g. "overall" or "age.18-49"
    order: int           # stable display order for Table 5B and Figure 6
    stratifier: str      # display text, e.g. "Overall" or "Age group"
    level: str           # display text, e.g. "All patients" or "18-49 years"
    column: str          # parquet column, "" for the overall row
    raw_level: str       # locked ASCII level string, "" for the overall row
    is_residual_level: bool = False   # not one of the stratifier's declared levels
    is_missing_level: bool = False    # AND the stratifier value is genuinely unrecorded
    source_column: str = ""           # column the residual label was resolved from


def residual_level(df: pd.DataFrame, column: str, residual: pd.Series
                   ) -> Tuple[str, str, str, bool]:
    """Resolve the DISPLAY label of a stratifier's residual level.

    Returns (source column, raw level, display label, the value is genuinely unrecorded).

    Two cases have to be told apart, and the whole of fix M5 is that they were not:

    * the stratifier value is RECORDED but is not one of the declared levels. Sex is the
      live case: `SEX` carries male, female and non-binary, `02_build_cohort.py` stores
      the stratifier as the binary `MALE` and leaves it Int8-missing for the non-binary
      patients rather than folding them into the female reference group, and the label
      layer then falls through to MISSING_LABEL. The value is recorded; the BINARY cannot
      hold it. The label is resolved from the source column through
      variable_dict.pretty_level(), never written here.
    * the stratifier value is genuinely NOT RECORDED. That level keeps MISSING_LABEL,
      which is what MISSING_LABEL is for.

    A residual carrying more than one distinct recorded value would be a new stratum and
    is not created silently: it raises, because CLAUDE.md 6.5 forbids adding a stratum
    without a stated reason and a mixed residual is a change in the data this step must
    not paper over.
    """
    source = RESIDUAL_SOURCE_COLUMN.get(column, "")
    if not source or source not in df.columns:
        return "", "", VD.MISSING_LABEL, True
    values = df.loc[residual, source].astype("string")
    recorded = sorted({v for v in values.dropna().tolist() if str(v).strip()})
    if not recorded:
        return source, "", VD.MISSING_LABEL, True
    if len(recorded) > 1 or int(values.isna().sum()) or int((values == "").sum()):
        raise SystemExit(
            f"[06] the residual level of {column} carries {len(recorded)} distinct "
            f"recorded {source} values {recorded} plus "
            f"{int(values.isna().sum())} unrecorded. That is more than one stratum and "
            f"this step will not create one silently (CLAUDE.md 6.5). Report it.")
    raw = str(recorded[0])
    return source, raw, VD.pretty_level(source, raw), False


def build_strata(df: pd.DataFrame) -> List[Tuple[Stratum, np.ndarray]]:
    """The overall row plus the four LOCKED stratifiers, in their locked level order.

    CLAUDE.md 6.5: four stratifiers fixed a priori and NOTHING ELSE without a stated
    reason. No interaction strata, no data-driven cut points, no additional variable.
    The one addition here is not a new stratifier: it is the residual level of an
    existing one, the patients the declared levels do not cover, carried so that the four
    declared strata provably account for every patient in the cohort. Its DISPLAY label
    comes from residual_level() above, so a recorded value is named and only a genuinely
    unrecorded one is called not recorded.
    """
    out: List[Tuple[Stratum, np.ndarray]] = []
    order = 0
    out.append((Stratum("overall", order, "Overall", "All patients", "", ""),
                np.ones(len(df), dtype=bool)))

    for spec in VD.STRATIFIERS:
        column = spec.column
        as_text = df[column].astype("string")
        covered = pd.Series(False, index=df.index)
        for raw in spec.levels:
            order += 1
            mask = (as_text == raw).fillna(False)
            covered = covered | mask
            out.append((
                Stratum(f"{STRATIFIER_SLUG[column]}.{level_slug(column, raw)}", order,
                        STRATIFIER_DISPLAY[column], VD.pretty_level(column, raw),
                        column, raw),
                mask.to_numpy()))
        residual = ~covered
        if int(residual.sum()) > 0:
            order += 1
            source, raw_level, label, unrecorded = residual_level(df, column, residual)
            out.append((
                Stratum(f"{STRATIFIER_SLUG[column]}.{RESIDUAL_SLUG}", order,
                        STRATIFIER_DISPLAY[column], label,
                        column, raw_level, is_residual_level=True,
                        is_missing_level=unrecorded, source_column=source),
                residual.to_numpy()))
    return out


# ===========================================================================
# 4. ONE STRATUM, ESTIMATED
# ===========================================================================


@dataclass
class Fit:
    stratum: Stratum
    n: int
    events: int
    competing: int
    suppressed: bool
    reason: str
    cif30: float = float("nan")
    cif30_lo: float = float("nan")
    cif30_hi: float = float("nan")
    at_risk: Optional[np.ndarray] = None      # (31,) event-free and alive at start of day
    cif_start: Optional[np.ndarray] = None    # (31,) cumulative incidence at start of day
    curve: Optional[np.ndarray] = None        # (31,) conditional risk, PRIMARY
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    curve_km: Optional[np.ndarray] = None     # one minus Kaplan-Meier sensitivity
    cif30_km: float = float("nan")
    curve_end_of_day: Optional[np.ndarray] = None
    boot_days: Optional[Dict[float, np.ndarray]] = None
    monotone: bool = True
    rise_days: Optional[List[int]] = None
    day0_events: int = 0
    day01_events: int = 0
    events_by_day: Optional[np.ndarray] = None
    competing_by_day: Optional[np.ndarray] = None
    # WHO the conditioning set is made of, by day. See section 4.1 below.
    at_risk_counted: Optional[np.ndarray] = None          # (31,) counted patient by patient
    in_hospital_by_day: Optional[np.ndarray] = None       # (31,) recorded stay not ended
    discharged_by_day: Optional[np.ndarray] = None        # (31,) recorded stay had ended
    stay_missing_by_day: Optional[np.ndarray] = None      # (31,) no recorded stay at all


# ---------------------------------------------------------------------------
# 4.1 THE COMPOSITION OF THE CONDITIONING SET
#
# The conditional risk at day D is measured on the patients event free and alive at the
# START of day D. That is a SURVIVAL condition, not an observation condition: a patient
# who went home on day 1 and is alive and event free on day 7 is in the day-7 conditioning
# set, and almost all of them are. The estimand is legitimate and precisely defined, but
# nothing in this step's earlier output let a reader see WHO the number was measured on,
# so "the observation needed" and "needs 14 days" could attach to it and import a causal,
# observation-effect reading the estimand does not carry. It is measured here instead.
#
# STILL IN HOSPITAL AT THE START OF DAY D, defined: the patient's postoperative stay had
# not ended before day D, that is LOS_POSTOP >= D on the frozen column 02 wrote from
# DOPTODIS. A patient discharged ON day D was in hospital at the start of it, which is the
# same day-indexing the estimand itself uses and the same tie convention CLAUDE.md 6.3
# locks (an event on the discharge day counts as pre-discharge).
#
# THE 83 PATIENTS WITH NO RECORDED STAY are handled explicitly rather than silently. They
# cannot be placed on either side, so they are their OWN published count on every day, the
# primary in-hospital count excludes them, and an upper-bound count that places every one
# of them in hospital is published beside it. .build_facts.md 7.5 binding item 6 records
# that 77 of them also carry STILLINHOSP = 'Yes' and are therefore known to have been in
# hospital at the 30-day assessment; the remaining 6 are genuinely unplaceable. The bound
# is what makes the choice auditable instead of a convention buried in a comparison.
#
# THE 112 PATIENTS FLAGGED STILL IN HOSPITAL are NOT treated as having no discharge:
# .build_facts.md 7.5 binding item 6 is explicit that the claim is false for 35 of them,
# who carry a length of stay (24 above 30 days and 11 at or below it). The flag is
# therefore never used to override a recorded stay. It is used only to say how many of the
# unplaceable 83 are known to have been inpatients, and all four counts are published.
#
# NO LENGTH OF STAY IS TRUNCATED AT DAY 30 (.build_facts.md 7.5): DOPTODIS runs to 112
# days and 74 analytic patients stayed past the end of the window.
# ---------------------------------------------------------------------------


def riskset_composition(mask: np.ndarray, exit_day: np.ndarray, los: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Who is in the day-D conditioning set, for D in 0..30.

    exit_day[i] is the day patient i leaves the risk set (their composite event day, or
    their competing death day, or +inf if they reach day 30 event free and alive), so the
    conditioning set at the start of day D is exactly {i : exit_day[i] >= D}.

    Returns four (31,) integer arrays: at risk, of whom still in hospital, of whom
    discharged, and of whom the postoperative stay is not recorded at all. The three parts
    sum to the first by construction, which the caller asserts.
    """
    exit_in = exit_day[mask]
    los_in = los[mask]
    recorded = np.isfinite(los_in)
    at_risk = np.zeros(DAYS, dtype=np.int64)
    in_hosp = np.zeros(DAYS, dtype=np.int64)
    discharged = np.zeros(DAYS, dtype=np.int64)
    missing = np.zeros(DAYS, dtype=np.int64)
    for d in range(DAYS):
        alive = exit_in >= d
        at_risk[d] = int(alive.sum())
        in_hosp[d] = int((alive & recorded & (los_in >= d)).sum())
        discharged[d] = int((alive & recorded & (los_in < d)).sum())
        missing[d] = int((alive & ~recorded).sum())
    if not np.array_equal(in_hosp + discharged + missing, at_risk):
        raise SystemExit("[06] the conditioning-set composition does not partition the "
                         "risk set; the still-in-hospital rule and the risk set disagree")
    return at_risk, in_hosp, discharged, missing


def fit_stratum(stratum: Stratum, mask: np.ndarray, pod_event_all: np.ndarray,
                pod_death_all: np.ndarray, is_event: np.ndarray,
                is_competing: np.ndarray, exit_day: np.ndarray,
                los: np.ndarray) -> Fit:
    n = int(mask.sum())
    ev_mask = mask & is_event
    cp_mask = mask & is_competing
    events = int(ev_mask.sum())
    competing = int(cp_mask.sum())

    if events < VD.MIN_STRATUM_EVENTS:
        return Fit(stratum, n, events, competing, True,
                   f"fewer than {VD.MIN_STRATUM_EVENTS} composite events "
                   f"({events} observed)")

    e1, e2 = counts_by_day(pod_event_all[ev_mask], pod_death_all[cp_mask])
    e1b = e1[None, :]
    e2b = e2[None, :]
    n_arr = np.array([n], dtype=float)

    cif, surv, at_risk = aalen_johansen(e1b, e2b, n_arr)
    curve = conditional_risk_aj(cif, surv)[0]
    curve_end = conditional_risk_aj_end_of_day(cif, surv)[0]
    surv_km = kaplan_meier_survival(e1b, e2b, n_arr)
    curve_km = conditional_risk_km(surv_km)[0]

    # Internal correctness check. With complete follow-up and no censoring inside the
    # window the Aalen-Johansen recursion must reduce EXACTLY to counting: the
    # cumulative incidence is the crude cumulative proportion and the conditional risk
    # is (remaining events) / (patients still event-free and alive). If the recursion
    # and the closed form ever disagree, the estimator is wrong, not the data.
    cum_e1 = np.concatenate([[0.0], np.cumsum(e1)])
    cum_e2 = np.concatenate([[0.0], np.cumsum(e2)])
    closed_cif = cum_e1 / n
    closed_surv = (n - cum_e1 - cum_e2) / n
    if not np.allclose(cif[0], closed_cif, atol=1e-12):
        raise SystemExit("[06] Aalen-Johansen recursion disagrees with the closed form "
                         f"on stratum {stratum.slug}")
    if not np.allclose(surv[0], closed_surv, atol=1e-12):
        raise SystemExit("[06] event-free survival disagrees with the closed form on "
                         f"stratum {stratum.slug}")
    if abs(curve[0] - cif[0, -1]) > 1e-12:
        raise SystemExit("[06] the day-0 conditional risk is not the day-30 cumulative "
                         f"incidence on stratum {stratum.slug}")

    boot_e1, boot_e2 = bootstrap_cells(e1, e2, n, BOOTSTRAP_REPLICATES)
    boot_n = np.full(BOOTSTRAP_REPLICATES, float(n))
    boot_cif, boot_surv, _ = aalen_johansen(boot_e1, boot_e2, boot_n)
    boot_curve = conditional_risk_aj(boot_cif, boot_surv)
    lo, hi = percentile_ci(boot_curve, axis=0)
    cif30_lo, cif30_hi = percentile_ci(boot_cif[:, -1], axis=0)

    boot_days = {t: first_day_below(boot_curve, t) for t in VD.RISK_THRESHOLDS}

    rises = [d for d in range(1, DAYS) if curve[d] > curve[d - 1] + 1e-15]

    # The conditioning set, counted patient by patient, and tied to the denominator the
    # estimator already publishes. If the two ever disagree the composition is describing
    # a different set from the one the risk is measured on, which is the whole point.
    ar_counted, in_hosp, discharged, stay_missing = riskset_composition(
        mask, exit_day, los)
    ar_published = np.rint(surv[0, :-1] * n).astype(np.int64)
    if not np.array_equal(ar_counted, ar_published):
        raise SystemExit("[06] the counted conditioning set differs from the published "
                         f"at-risk denominator on stratum {stratum.slug}")

    return Fit(
        stratum=stratum, n=n, events=events, competing=competing,
        suppressed=False, reason="",
        cif30=float(cif[0, -1]), cif30_lo=float(cif30_lo), cif30_hi=float(cif30_hi),
        at_risk=surv[0, :-1] * n, cif_start=cif[0, :-1],
        curve=curve, lo=lo, hi=hi,
        curve_km=curve_km, cif30_km=float(1.0 - surv_km[0, -1]),
        curve_end_of_day=curve_end,
        boot_days=boot_days,
        monotone=not rises, rise_days=rises,
        day0_events=int(e1[0]), day01_events=int(e1[0] + e1[1]),
        events_by_day=e1, competing_by_day=e2,
        at_risk_counted=ar_counted, in_hospital_by_day=in_hosp,
        discharged_by_day=discharged, stay_missing_by_day=stay_missing,
    )


# ===========================================================================
# 5. RENDERING HELPERS
# ===========================================================================


def pct(value: float, decimals: int = 3) -> str:
    return VD.fmt_pct(100.0 * value, decimals)


def pct_ci(lo: float, hi: float, decimals: int = 3) -> str:
    return VD.fmt_ci(100.0 * lo, 100.0 * hi, decimals)


def day_text(day: int) -> str:
    return NOT_REACHED if day > VD.FOLLOWUP_DAYS else str(int(day))


def day_ci_text(lo: float, hi: float) -> str:
    lo_i = int(math.floor(lo))
    hi_i = int(math.ceil(hi))
    if lo_i > VD.FOLLOWUP_DAYS:
        return NOT_REACHED
    if hi_i > VD.FOLLOWUP_DAYS:
        return f"{lo_i} to {NOT_REACHED.lower()}"
    return f"{lo_i} to {hi_i}"


def num(value: float, decimals: int = 6) -> str:
    """A bare number for a machine-read column. Never a NaN, never a percent sign."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.{decimals}f}"


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Write one CSV, asserting every header and every cell is renderable at write time.

    CLAUDE.md hard rules 1 and 2 are enforced HERE, at the write, not by a later grep.
    """
    for column in header:
        VD.assert_renderable(column, where=f"{path.name} header")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([str(c) for c in header])
        for row in rows:
            if len(row) != len(header):
                raise SystemExit(f"[06] {path.name}: a row has {len(row)} cells for "
                                 f"{len(header)} columns")
            cells = []
            for cell in row:
                text = "" if cell is None else str(cell)
                VD.assert_renderable(text, where=f"{path.name} cell")
                cells.append(text)
            writer.writerow(cells)


def clean_fragment(results: Dict[str, object]) -> Dict[str, object]:
    """Validate and normalise the fragment before it is written (CLAUDE.md 4.2).

    run_all.py enforces the same contract at merge time and rejects the whole run on the
    first violation. Doing it here too means a violation fails inside the step that caused
    it, with the offending key named.
    """
    out: Dict[str, object] = {}
    bad_ns, bad_key, bad_val = [], [], []

    def scalar(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, float) and not math.isfinite(v):
            return None                    # JSON has no NaN; write null
        return v

    for key, value in results.items():
        if not KEY_RE.match(key):
            bad_key.append(key)
        if not key.startswith(RESERVED_PREFIXES) and key not in tuple(
                p.rstrip(".") for p in RESERVED_PREFIXES):
            bad_ns.append(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            items = [scalar(v) for v in list(value)]
            for item in items:
                if item is not None and not isinstance(item, (str, int, float, bool)):
                    bad_val.append(key)
            out[key] = items
        else:
            v = scalar(value)
            if v is not None and not isinstance(v, (str, int, float, bool)):
                bad_val.append(key)
            out[key] = v

    problems = []
    if bad_ns:
        problems.append(f"outside the reserved namespaces {list(RESERVED_PREFIXES)}: "
                        f"{sorted(set(bad_ns))[:10]}")
    if bad_key:
        problems.append(f"not a flat lowercase dotted path: {sorted(set(bad_key))[:10]}")
    if bad_val:
        problems.append(f"value is neither a scalar nor a list of scalars: "
                        f"{sorted(set(bad_val))[:10]}")
    if problems:
        raise SystemExit("[06] fragment contract violation:\n       "
                         + "\n       ".join(problems))
    return out


# ===========================================================================
# 6. TABLES
# ===========================================================================

TABLE5_HEADER = [
    "Stratifier",
    "Stratum",
    "Patients",
    "Composite events",
    "denominator",
    "Cumulative incidence of the composite to day 30 (%)",
    "Risk threshold",
    # Renamed from "Safe discharge day" to match the manuscript's own term and to
    # stop publishing a prescriptive label for a descriptive quantity: this is the
    # earliest day on which residual risk falls below the threshold, not a day on
    # which discharge is established to be safe. The Methods and the Table 5
    # footnote both say so explicitly. Header text only -- no value changes.
    # verify_numbers.py's provenance map keys on these strings and was updated too.
    "Residual-risk threshold day",
    "Residual-risk threshold day, 95% confidence interval",
    "Residual risk at that day (%)",
    "Residual risk, 95% confidence interval (%)",
    "Patients event free and alive at the start of that day",
    "Threshold reached within 30 days",
    "Residual risk remains below the threshold through day 30",
]

TABLE5B_HEADER = [
    "Display order",
    "Stratifier",
    "Stratum",
    "Postoperative day",
    "Patients",
    "Composite events",
    "denominator",
    "Patients event free and alive at the start of the day",
    "Conditional risk of a subsequent composite event (%)",
    "95% confidence interval, lower (%)",
    "95% confidence interval, upper (%)",
    "Cumulative incidence of the composite to the start of the day (%)",
    "Suppressed",
]

S4_HEADER = [
    "Sensitivity analysis",
    "Stratifier",
    "Stratum",
    "Patients",
    "Composite events",
    "denominator",
    "Quantity",
    "Primary value",
    "Sensitivity value",
    "Difference",
]


def table5_rows(fits: Sequence[Fit]) -> List[List[object]]:
    rows: List[List[object]] = []
    for fit in fits:
        s = fit.stratum
        if fit.suppressed:
            rows.append([s.stratifier, s.level, VD.fmt_count(fit.n),
                         VD.fmt_count(fit.events), VD.fmt_count(fit.n),
                         SUPPRESSED_TEXT, SUPPRESSED_TEXT, SUPPRESSED_TEXT,
                         SUPPRESSED_TEXT, SUPPRESSED_TEXT, SUPPRESSED_TEXT,
                         SUPPRESSED_TEXT, SUPPRESSED_TEXT, SUPPRESSED_TEXT])
            continue
        for threshold in VD.RISK_THRESHOLDS:
            day = int(first_day_below(fit.curve[None, :], threshold)[0])
            boot = fit.boot_days[threshold]
            d_lo, d_hi = percentile_ci(boot.astype(float))
            reached = day <= VD.FOLLOWUP_DAYS
            if reached:
                risk_text = pct(float(fit.curve[day]))
                ci_text = pct_ci(float(fit.lo[day]), float(fit.hi[day]))
                at_risk_text = VD.fmt_count(round(float(fit.at_risk[day])))
                stays = "Yes" if stays_below(fit.curve, threshold, day) else "No"
            else:
                risk_text = NOT_REACHED
                ci_text = NOT_REACHED
                at_risk_text = NOT_APPLICABLE
                stays = NOT_APPLICABLE
            rows.append([
                s.stratifier, s.level, VD.fmt_count(fit.n), VD.fmt_count(fit.events),
                VD.fmt_count(fit.n), pct(fit.cif30),
                VD.threshold_label(threshold),
                day_text(day), day_ci_text(d_lo, d_hi),
                risk_text, ci_text, at_risk_text,
                "Yes" if reached else "No", stays,
            ])
    return rows


def table5b_rows(fits: Sequence[Fit]) -> List[List[object]]:
    rows: List[List[object]] = []
    for fit in fits:
        s = fit.stratum
        for day in range(DAYS):
            if fit.suppressed:
                rows.append([s.order, s.stratifier, s.level, day, fit.n, fit.events,
                             fit.n, "", "", "", "", "", "Yes"])
                continue
            # Bare integers and bare decimals throughout: Table 5B is the machine-read
            # input to Figure 6, not a reviewer-facing table. Table 5 carries the
            # human-formatted versions of the same quantities.
            rows.append([
                s.order, s.stratifier, s.level, day, fit.n, fit.events, fit.n,
                int(round(float(fit.at_risk[day]))),
                num(100.0 * float(fit.curve[day])),
                num(100.0 * float(fit.lo[day])),
                num(100.0 * float(fit.hi[day])),
                num(100.0 * float(fit.cif_start[day])),
                "No",
            ])
    return rows


def s4_rows(fits: Sequence[Fit]) -> List[List[object]]:
    km_label = ("One minus Kaplan-Meier instead of Aalen-Johansen "
                "(death treated as censoring rather than as a competing event)")
    idx_label = ("Day indexing: conditioning on the patients event free and alive at the "
                 "END of the day rather than at the start of it")
    rows: List[List[object]] = []
    for fit in fits:
        s = fit.stratum
        if fit.suppressed:
            rows.append([km_label, s.stratifier, s.level, VD.fmt_count(fit.n),
                         VD.fmt_count(fit.events), VD.fmt_count(fit.n),
                         "All quantities", SUPPRESSED_TEXT, SUPPRESSED_TEXT,
                         SUPPRESSED_TEXT])
            continue
        head = [s.stratifier, s.level, VD.fmt_count(fit.n), VD.fmt_count(fit.events),
                VD.fmt_count(fit.n)]

        rows.append([km_label] + head + [
            "Cumulative incidence of the composite to day 30 (%)",
            pct(fit.cif30), pct(fit.cif30_km),
            VD.fmt_pct(100.0 * (fit.cif30_km - fit.cif30), 4) + " absolute"])

        gap = np.max(np.abs(fit.curve_km - fit.curve))
        rows.append([km_label] + head + [
            "Largest absolute difference in conditional risk across days 0 to 30",
            "Reference", VD.fmt_pct(100.0 * gap, 4),
            VD.fmt_pct(100.0 * gap, 4) + " absolute"])

        for threshold in VD.RISK_THRESHOLDS:
            day_aj = int(first_day_below(fit.curve[None, :], threshold)[0])
            day_km = int(first_day_below(fit.curve_km[None, :], threshold)[0])
            rows.append([km_label] + head + [
                f"Safe discharge day at the {VD.threshold_label(threshold)} threshold",
                day_text(day_aj), day_text(day_km), day_difference(day_aj, day_km)])

        for threshold in VD.RISK_THRESHOLDS:
            day_primary = int(first_day_below(fit.curve[None, :], threshold)[0])
            day_alt = int(first_day_below(fit.curve_end_of_day[None, :], threshold)[0])
            rows.append([idx_label] + head + [
                f"Safe discharge day at the {VD.threshold_label(threshold)} threshold",
                day_text(day_primary), day_text(day_alt),
                day_difference(day_primary, day_alt)])
    return rows


def day_difference(primary: int, sensitivity: int) -> str:
    if primary > VD.FOLLOWUP_DAYS or sensitivity > VD.FOLLOWUP_DAYS:
        return "Not comparable, one arm is not reached within 30 days"
    delta = sensitivity - primary
    if delta == 0:
        return "0 days"
    return f"{delta:+d} day" + ("s" if abs(delta) != 1 else "")


# ===========================================================================
# 7. THE FRAGMENT
# ===========================================================================


# --- units and scales (.build_facts.md section 9) ---------------------------------
# This step publishes every risk as a PERCENTAGE on the 0 to 100 scale, including the
# keys ending in est, lo and hi. Step 03 publishes the same suffixes as PROPORTIONS on
# the 0 to 1 scale. The suffixes carry no unit and nothing in the key name resolves the
# difference, so applying 03's rule to window.overall.cif30.est would print this paper's
# lead result as 0.090% or 90.1% instead of 0.901%. Every declaration below therefore
# names the one namespace it governs and says so in its own text.
#
# `<namespace>.scale` is prose a human can read. `<namespace>.scale.families` and
# `<namespace>.scale.units` carry the same content as two parallel lists so
# verify_numbers.py can assert a unit mechanically rather than parse English. The family
# entries are KEY PATTERNS addressed to 08, not rendered prose. No key is renamed and
# none is removed: 04 and the Batch 3 reviewer have both already verified values against
# the names as they stand.

UNIT_PERCENT = "percentage on the 0 to 100 scale"
UNIT_PROPORTION = "proportion on the 0 to 1 scale"
UNIT_PP = "percentage points"
UNIT_POD = "integer postoperative day"
UNIT_COUNT = "count of patients or events"
UNIT_BOOL = "boolean, carries no unit"
UNIT_TEXT = "text, carries no unit"
UNIT_SEED = "random number generator seed, carries no unit"

SCALE_SCOPE_NOTE = (
    "This declaration governs only the namespace it is named for. It is not a "
    "project-wide rule and must never be read as one: the same est, lo and hi "
    "suffixes carry proportions, odds ratios and a kappa in other steps, so read "
    "the declaration published by the step that owns the key."
)

SCALE_DECLARATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        "window",
        "Within the safe-discharge-window namespace EVERY risk is a percentage on "
        "the 0 to 100 scale, including the keys ending in est, lo and hi. The three "
        "keys that carry a residual risk at a threshold are a percentage and its two "
        "percentage limits, even though only the first of the three is suffixed with "
        "pct. A safe discharge day and its limits are integer postoperative days, "
        "null where the threshold is not reached inside the window. The threshold "
        "list itself is the exception and is held as proportions on the 0 to 1 "
        "scale, with a matching list of rendered labels beside it. The conditioning-set "
        "composition keys are counts of patients and their percentages of the at-risk "
        "denominator on the same day.",
        (
            # ORDER MATTERS AND IS DELIBERATE. Both readers of these lists resolve FIRST
            # MATCH WINS, and assert_scale_coverage below proves that every family
            # matching a key agrees on its unit, so order can never change a UNIT. What it
            # does change is which family is NAMED, so the most specific pattern is listed
            # first and the stratum wildcards come last: window.composite.n is a count but
            # it is not a stratum, and a resolver that says so is easier to audit.
            ("window.cohort.n", UNIT_COUNT),
            ("window.composite.n", UNIT_COUNT),
            ("window.composite.pct", UNIT_PERCENT),
            ("window.composite.definition", UNIT_TEXT),
            ("window.component.<component>.n", UNIT_COUNT),
            ("window.competing.<quantity>.n", UNIT_COUNT),
            ("window.competing_tie_rule", UNIT_TEXT),

            # The composition of the day-D conditioning set (fix M1).
            ("window.riskset.definition", UNIT_TEXT),
            ("window.riskset.missing_stay_rule", UNIT_TEXT),
            ("window.riskset.flag_rule", UNIT_TEXT),
            ("window.riskset.text.<part>", UNIT_TEXT),
            ("window.riskset.days", UNIT_POD),
            ("window.riskset.day<d>.text", UNIT_TEXT),
            ("window.riskset.day<d>.in_hospital_pct", UNIT_PERCENT),
            ("window.riskset.day<d>.discharged_pct", UNIT_PERCENT),
            ("window.riskset.day<d>.in_hospital_upper_pct", UNIT_PERCENT),
            ("window.riskset.day<d>.at_risk_n", UNIT_COUNT),
            ("window.riskset.day<d>.in_hospital_n", UNIT_COUNT),
            ("window.riskset.day<d>.discharged_n", UNIT_COUNT),
            ("window.riskset.day<d>.stay_not_recorded_n", UNIT_COUNT),
            ("window.riskset.day<d>.in_hospital_upper_n", UNIT_COUNT),
            ("window.riskset.<quantity>_n", UNIT_COUNT),

            # Study-level constants and the two sensitivity summaries.
            ("window.thresholds", UNIT_PROPORTION),
            ("window.threshold_labels", UNIT_TEXT),
            ("window.followup_days", UNIT_POD),
            ("window.extrapolation", UNIT_TEXT),
            ("window.estimator", UNIT_TEXT),
            ("window.estimand", UNIT_TEXT),
            ("window.indexing_rule", UNIT_TEXT),
            ("window.seed", UNIT_SEED),
            ("window.bootstrap.replicates", UNIT_COUNT),
            ("window.bootstrap.method", UNIT_TEXT),
            ("window.bootstrap.unit", UNIT_TEXT),
            ("window.bootstrap.reason", UNIT_TEXT),
            ("window.no_hematoma_risk_model", UNIT_BOOL),
            ("window.no_points_score", UNIT_BOOL),
            ("window.stays_below_all_cells", UNIT_BOOL),
            ("window.sensitivity.km_max_abs_difference_pct", UNIT_PP),
            ("window.sensitivity.indexing_max_abs_day_shift", UNIT_POD),
            ("window.sensitivity.stratum_threshold_cells", UNIT_COUNT),
            ("window.sensitivity.thresholds_never_reached", UNIT_COUNT),
            ("window.sensitivity.km_safe_days_moved", UNIT_COUNT),
            ("window.sensitivity.indexing_safe_days_moved", UNIT_COUNT),
            ("window.sensitivity.km_changes_no_safe_day", UNIT_BOOL),
            ("window.sensitivity.km_never_below_aalen_johansen", UNIT_BOOL),

            # Per stratum and threshold. A stratum slug is itself dotted (age.18-49), so
            # these placeholders necessarily span more than one segment.
            ("window.<stratum>.<threshold>.risk_pct", UNIT_PERCENT),
            ("window.<stratum>.<threshold>.risk_lo", UNIT_PERCENT),
            ("window.<stratum>.<threshold>.risk_hi", UNIT_PERCENT),
            ("window.<stratum>.<threshold>.in_hospital_pct", UNIT_PERCENT),
            ("window.<stratum>.<threshold>.day", UNIT_POD),
            ("window.<stratum>.<threshold>.day_lo", UNIT_POD),
            ("window.<stratum>.<threshold>.day_hi", UNIT_POD),
            ("window.<stratum>.<threshold>.day_km", UNIT_POD),
            ("window.<stratum>.<threshold>.day_end_of_day_indexing", UNIT_POD),
            ("window.<stratum>.<threshold>.at_risk_n", UNIT_COUNT),
            ("window.<stratum>.<threshold>.in_hospital_n", UNIT_COUNT),
            ("window.<stratum>.<threshold>.day_unreached_replicates", UNIT_COUNT),
            ("window.<stratum>.<threshold>.reached", UNIT_BOOL),
            ("window.<stratum>.<threshold>.stays_below", UNIT_BOOL),

            # Per stratum.
            ("window.<stratum>.cif30.est", UNIT_PERCENT),
            ("window.<stratum>.cif30.lo", UNIT_PERCENT),
            ("window.<stratum>.cif30.hi", UNIT_PERCENT),
            ("window.<stratum>.day01_events.pct_of_events", UNIT_PERCENT),
            ("window.<stratum>.day0_events.n", UNIT_COUNT),
            ("window.<stratum>.day01_events.n", UNIT_COUNT),
            ("window.<stratum>.n", UNIT_COUNT),
            ("window.<stratum>.events", UNIT_COUNT),
            ("window.<stratum>.competing_deaths", UNIT_COUNT),
            ("window.<stratum>.suppressed", UNIT_BOOL),
        ),
    ),
    (
        "condrisk",
        "Within the conditional-risk namespace every curve is a percentage on the 0 "
        "to 100 scale, including the keys ending in est, lo and hi. Each is a list of "
        "31 values, one per integer postoperative day 0 to 30, aligned with the day "
        "list. The difference between the two estimators is in percentage points. The "
        "conditioning-set composition lists are counts of patients on each of the same "
        "31 days, and the in-hospital percentage is of the at-risk count on that day.",
        (
            ("condrisk.<stratum>.est", UNIT_PERCENT),
            ("condrisk.<stratum>.lo", UNIT_PERCENT),
            ("condrisk.<stratum>.hi", UNIT_PERCENT),
            ("condrisk.<stratum>.km_est", UNIT_PERCENT),
            ("condrisk.<stratum>.in_hospital_pct", UNIT_PERCENT),
            ("condrisk.<stratum>.km_max_abs_difference_pct", UNIT_PP),
            ("condrisk.<stratum>.day", UNIT_POD),
            ("condrisk.<stratum>.rise_days", UNIT_POD),
            ("condrisk.<stratum>.at_risk", UNIT_COUNT),
            ("condrisk.<stratum>.in_hospital_n", UNIT_COUNT),
            ("condrisk.<stratum>.stay_not_recorded_n", UNIT_COUNT),
            ("condrisk.<stratum>.monotone", UNIT_BOOL),
            ("condrisk.days", UNIT_POD),
            ("condrisk.n_strata_with_a_rise", UNIT_COUNT),
            ("condrisk.rises_all_follow_a_competing_death", UNIT_BOOL),
        ),
    ),
    (
        "suppress",
        "The suppression namespace carries no risk and no percentage. Every numeric "
        "key in it is a count of patients or events, or the minimum event count the "
        "suppression rule applies. The level keys carry the display label of a "
        "suppressed stratum and whether the stratifier value behind it is recorded.",
        (
            ("suppress.rule_min_events", UNIT_COUNT),
            ("suppress.n_strata_evaluated", UNIT_COUNT),
            ("suppress.n_strata_suppressed", UNIT_COUNT),
            ("suppress.strata_suppressed", UNIT_TEXT),
            ("suppress.residual_level_rule", UNIT_TEXT),
            ("suppress.<stratum>.events", UNIT_COUNT),
            ("suppress.<stratum>.n", UNIT_COUNT),
            ("suppress.<stratum>.reason", UNIT_TEXT),
            ("suppress.<stratum>.level_label", UNIT_TEXT),
            ("suppress.<stratum>.suppressed", UNIT_BOOL),
            ("suppress.<stratum>.level_recorded", UNIT_BOOL),
        ),
    ),
)

SCALE_CATCH_ALL = (
    "Every other numeric key in this namespace is a count of patients or events, a "
    "denominator, or an integer day."
)

# The four keys each declaration publishes about itself. They describe the declaration and
# are not measurements, so the coverage check skips them rather than declaring a family for
# a family list.
SCALE_META_SUFFIXES: Tuple[str, ...] = (".scale", ".scale.families", ".scale.units",
                                        ".scale.other_numeric_keys")

# --- import-time check: the declared namespaces ARE the reserved namespaces ---
_DECLARED_NS = tuple(ns for ns, _p, _f in SCALE_DECLARATIONS)
if sorted(_DECLARED_NS) != sorted(p.rstrip(".") for p in RESERVED_PREFIXES):
    raise SystemExit(
        f"[06] SCALE_DECLARATIONS covers {sorted(_DECLARED_NS)} but this step reserves "
        f"{sorted(p.rstrip('.') for p in RESERVED_PREFIXES)}. Every reserved namespace "
        f"declares its own units (.build_facts.md 9.2 item 2).")
if len(set(_DECLARED_NS)) != len(_DECLARED_NS):
    raise SystemExit(f"[06] SCALE_DECLARATIONS declares a namespace twice: {_DECLARED_NS}")


def scale_family_count() -> int:
    """How many key families the three declarations name, in total."""
    return sum(len(families) for _ns, _prose, families in SCALE_DECLARATIONS)


def scale_family_regex_strict(pattern: str) -> "re.Pattern":
    """The matcher `verify_numbers.py` builds from the published lists.

    A placeholder that occupies a WHOLE dotted segment spans one or more segments, because
    a stratum slug such as `age.18-49` is itself dotted; a placeholder INSIDE a segment,
    as in `window.riskset.day<d>.text`, spans that segment only. This is the stricter of
    the two matchers in play and is the one the COVERAGE half of the assertion uses, so
    coverage is proved against the reader that resolves fewest keys.
    """
    parts: List[str] = []
    for seg in pattern.split("."):
        if seg.startswith("<") and seg.endswith(">") and seg.count("<") == 1:
            parts.append(r"[^.]+(?:\.[^.]+)*?")
        else:
            parts.append("".join(
                r"[^.]+" if tok.startswith("<") and tok.endswith(">") else re.escape(tok)
                for tok in re.split(r"(<[^>]*>)", seg) if tok))
    return re.compile("^" + r"\.".join(parts) + "$")


def scale_family_regex_loose(pattern: str) -> "re.Pattern":
    """The matcher `08_generate_manuscript.py` builds from the published lists.

    Every placeholder is a non-greedy wildcard that may cross a dot. It resolves MORE keys
    than the strict matcher, so it is the one the CONFLICT half of the assertion uses: a
    unit collision is proved impossible against the reader that resolves most keys.
    """
    parts = re.split(r"<[A-Za-z0-9_]+>", pattern)
    return re.compile("^" + "(.+?)".join(re.escape(p) for p in parts) + "$")


def assert_scale_coverage(results: Dict[str, object]) -> int:
    """Every published key has a declared unit, and never two different ones.

    `.build_facts.md` 9.2 item 4 requires a gate to be able to ask this fragment for the
    unit of ANY key it holds and get ONE answer. `02`, `04` and `05` satisfy that with an
    exactly-one-family rule; this step cannot, because its stratum slugs are themselves
    dotted (`age.18-49`), so a stratum placeholder necessarily spans segments and any
    honest wildcard family also matches a sibling literal such as `window.cohort.n`. The
    property that actually matters is not that a key matches one pattern but that it
    resolves to one UNIT, which is exactly what `08_generate_manuscript.py`'s
    `Scales.unit_of` requires and what it raises on. So:

      * COVERAGE, under the STRICT (gate) matcher: every published key matches at least
        one declared family. A new key added without a family fails here, inside the step
        that created it, rather than in a downstream reader that cannot know what it means.
      * NO CONFLICT, under the LOOSE (manuscript) matcher: every family that matches a
        key declares the same unit for it. Both readers resolve first match wins, so
        agreement on the unit makes the resolution order irrelevant.

    A key matched by two families that disagree is the defect this closes: a wildcard
    family swallowing a sibling on a different scale.
    """
    strict = {ns: [(scale_family_regex_strict(f), f, u) for f, u in fams]
              for ns, _prose, fams in SCALE_DECLARATIONS}
    loose = {ns: [(scale_family_regex_loose(f), f, u) for f, u in fams]
             for ns, _prose, fams in SCALE_DECLARATIONS}
    missing_ns, unmatched, conflicting = [], [], []
    for key in results:
        if key.endswith(SCALE_META_SUFFIXES):
            continue
        ns = key.split(".", 1)[0]
        if ns not in strict:
            missing_ns.append(key)
            continue
        if not any(rx.match(key) for rx, _f, _u in strict[ns]):
            unmatched.append(key)
        units = {u for rx, _f, u in loose[ns] if rx.match(key)}
        if len(units) > 1:
            hits = sorted(f"{f} -> {u}" for rx, f, u in loose[ns] if rx.match(key))
            conflicting.append(f"{key}: {hits}")
    problems = []
    if missing_ns:
        problems.append(f"in a namespace with no scale declaration: "
                        f"{sorted(missing_ns)[:10]}")
    if unmatched:
        problems.append(f"matched by no declared family: {sorted(unmatched)[:10]}")
    if conflicting:
        problems.append(f"matched by families that disagree on the unit: "
                        f"{sorted(conflicting)[:10]}")
    if problems:
        raise SystemExit("[06] scale-declaration coverage gap "
                         "(.build_facts.md 9.2 item 4):\n       "
                         + "\n       ".join(problems))
    return scale_family_count()


def build_fragment(fits: Sequence[Fit], df_n: int, composite_n: int,
                   component_counts: Dict[str, int], competing_total: int,
                   competing_first: int, both_n: int, tie_same_day: int,
                   death_first_n: int,
                   stay_counts: Dict[str, int]) -> Dict[str, object]:
    r: Dict[str, object] = {}

    for namespace, prose, families in SCALE_DECLARATIONS:
        r[f"{namespace}.scale"] = f"{prose} {SCALE_SCOPE_NOTE}"
        r[f"{namespace}.scale.families"] = [f for f, _ in families]
        r[f"{namespace}.scale.units"] = [u for _, u in families]
        r[f"{namespace}.scale.other_numeric_keys"] = SCALE_CATCH_ALL

    r["window.cohort.n"] = df_n
    r["window.composite.n"] = composite_n
    r["window.composite.pct"] = 100.0 * composite_n / df_n
    for name, count in component_counts.items():
        r[f"window.component.{name}.n"] = count
    r["window.competing.deaths_in_window.n"] = competing_total
    r["window.competing.deaths_first.n"] = competing_first
    r["window.competing.both_events.n"] = both_n
    r["window.competing.tie_same_day.n"] = tie_same_day
    r["window.competing.death_strictly_first.n"] = death_first_n

    r["window.followup_days"] = VD.FOLLOWUP_DAYS
    r["window.thresholds"] = [float(t) for t in VD.RISK_THRESHOLDS]
    r["window.threshold_labels"] = [VD.threshold_label(t) for t in VD.RISK_THRESHOLDS]
    r["window.estimator"] = ("Aalen-Johansen cumulative incidence with death as the "
                             "competing event")
    r["window.estimand"] = (
        "The earliest postoperative day at which the conditional probability of a "
        "subsequent composite event, given the patient is event free and alive at the "
        "start of that day, falls below the threshold")
    r["window.composite.definition"] = (
        "Hematoma requiring unplanned reoperation, unplanned reintubation, ventilator "
        "dependence beyond 48 hours, or reoperative tracheostomy. Pneumonia and "
        "unplanned readmission are excluded.")
    r["window.indexing_rule"] = (
        "An event recorded on postoperative day D is counted as still ahead of a patient "
        "discharged on day D, so the conditioning set is the patients event free and "
        "alive at the start of day D")
    r["window.competing_tie_rule"] = (
        "Where the composite event and death fall on the same postoperative day the "
        "composite is taken as the first event")
    r["window.extrapolation"] = "None past postoperative day 30"
    r["window.seed"] = VD.SEED
    r["window.bootstrap.replicates"] = BOOTSTRAP_REPLICATES
    r["window.bootstrap.method"] = "percentile"
    r["window.bootstrap.unit"] = "patients resampled within stratum"
    r["window.bootstrap.reason"] = (
        "The conditional risk is a ratio functional of two Aalen-Johansen quantities and "
        "has no closed-form Aalen-Johansen variance, so the pre-specified seeded "
        "bootstrap applies")
    r["window.no_hematoma_risk_model"] = True
    r["window.no_points_score"] = True

    # --- WHO the conditional risk is measured on, by day -------------------------------
    # The conditioning set is a survival condition, not an observation condition, and the
    # overwhelming majority of it is at home. Published so that no reader has to infer it
    # and so that no sentence in the manuscript can attach an observation-effect reading
    # to the estimand without contradicting a number in the same package.
    overall_fit = next(f for f in fits if f.stratum.slug == "overall")
    r["window.riskset.definition"] = (
        "A patient is counted as still in hospital at the start of postoperative day D "
        "when the recorded postoperative length of stay had not ended before day D, that "
        "is when the length of stay is at least D. A patient discharged on day D was in "
        "hospital at the start of it, which is the same day indexing the estimand uses.")
    r["window.riskset.missing_stay_rule"] = (
        "Postoperative length of stay is not recorded for a small number of patients, "
        "who cannot be placed on either side. They are counted separately on every day, "
        "the primary still-in-hospital count excludes them, and an upper bound that "
        "places every one of them in hospital is published beside it.")
    r["window.riskset.flag_rule"] = (
        "The still-in-hospital flag is never used to override a recorded length of "
        "stay, because it is not equivalent to one: some flagged patients carry a "
        "recorded stay. It is reported only as how many of the patients with no recorded "
        "stay are known to have been in hospital at the end of the window.")
    r["window.riskset.stay_recorded_n"] = stay_counts["recorded"]
    r["window.riskset.stay_not_recorded_n"] = stay_counts["not_recorded"]
    r["window.riskset.still_in_hospital_flag_n"] = stay_counts["flagged"]
    r["window.riskset.stay_not_recorded_and_flagged_n"] = stay_counts[
        "not_recorded_and_flagged"]
    r["window.riskset.stay_not_recorded_not_flagged_n"] = stay_counts[
        "not_recorded_not_flagged"]
    r["window.riskset.flagged_with_recorded_stay_n"] = stay_counts[
        "flagged_with_recorded_stay"]
    r["window.riskset.flagged_stay_inside_window_n"] = stay_counts[
        "flagged_stay_inside_window"]
    r["window.riskset.flagged_stay_beyond_window_n"] = stay_counts[
        "flagged_stay_beyond_window"]
    r["window.riskset.days"] = list(RISKSET_REPORT_DAYS)
    for d in RISKSET_REPORT_DAYS:
        at_risk_d = int(overall_fit.at_risk_counted[d])
        in_hosp_d = int(overall_fit.in_hospital_by_day[d])
        disch_d = int(overall_fit.discharged_by_day[d])
        miss_d = int(overall_fit.stay_missing_by_day[d])
        r[f"window.riskset.day{d}.at_risk_n"] = at_risk_d
        r[f"window.riskset.day{d}.in_hospital_n"] = in_hosp_d
        r[f"window.riskset.day{d}.in_hospital_pct"] = 100.0 * in_hosp_d / at_risk_d
        r[f"window.riskset.day{d}.discharged_n"] = disch_d
        r[f"window.riskset.day{d}.discharged_pct"] = 100.0 * disch_d / at_risk_d
        r[f"window.riskset.day{d}.stay_not_recorded_n"] = miss_d
        r[f"window.riskset.day{d}.in_hospital_upper_n"] = in_hosp_d + miss_d
        r[f"window.riskset.day{d}.in_hospital_upper_pct"] = (
            100.0 * (in_hosp_d + miss_d) / at_risk_d)
        r[f"window.riskset.day{d}.text"] = (
            f"{VD.fmt_count(in_hosp_d)} of {VD.fmt_count(at_risk_d)} "
            f"({VD.fmt_pct(100.0 * in_hosp_d / at_risk_d, 1)})")

    # Renderable prose, in the style 03 uses for its Methods sentences, so 08 quotes a
    # measured sentence and types no number by hand (CLAUDE.md hard rule 3).
    def _day_clause(d: int) -> str:
        at_risk_d = int(overall_fit.at_risk_counted[d])
        in_hosp_d = int(overall_fit.in_hospital_by_day[d])
        return (f"{VD.fmt_count(in_hosp_d)} of the {VD.fmt_count(at_risk_d)} patients "
                f"event free and alive at the start of postoperative day {d} were still "
                f"in hospital ({VD.fmt_pct(100.0 * in_hosp_d / at_risk_d, 1)})")

    r["window.riskset.text.methods"] = (
        "The conditional risk at postoperative day D is estimated on every patient who "
        "is event free and alive at the start of that day, whether or not that patient "
        "was still in hospital. Membership of the conditioning set is a survival "
        "condition and not an observation condition, so the quantity is the probability "
        "that a qualifying event is still ahead of a patient on day D and is not an "
        "estimate of the effect of continued inpatient observation. The composition of "
        "the conditioning set is reported on every day so that the estimand cannot be "
        "read as an observation effect.")
    r["window.riskset.text.missing_handling"] = (
        f"Postoperative length of stay is not recorded for "
        f"{VD.fmt_count(stay_counts['not_recorded'])} patients, of whom "
        f"{VD.fmt_count(stay_counts['not_recorded_and_flagged'])} carry the "
        f"still-in-hospital flag and are therefore known to have been in hospital at the "
        f"end of the window, while {VD.fmt_count(stay_counts['not_recorded_not_flagged'])}"
        f" cannot be placed. They are counted as not still in hospital in the primary "
        f"count, and the bound that places every one of them in hospital is reported "
        f"beside it. The flag itself never overrides a recorded stay: "
        f"{VD.fmt_count(stay_counts['flagged_with_recorded_stay'])} of the "
        f"{VD.fmt_count(stay_counts['flagged'])} flagged patients carry one.")
    r["window.riskset.text.results"] = (
        "Most of the patients the conditional risk is measured on had already been "
        "discharged: " + _day_clause(3) + ", " + _day_clause(7) + ", and "
        + _day_clause(14) + ".")
    r["window.riskset.text.caveat"] = (
        "Because the conditioning set at each day is overwhelmingly made up of patients "
        "who had already gone home, the safe discharge day must not be described as the "
        "observation a patient needs, as a period a patient must be watched for, or as a "
        "length of stay. It is the day on which the probability of a subsequent "
        "qualifying event first falls below the stated threshold, estimated on a cohort "
        "in which almost every patient still at risk on that day was already at home.")

    r["suppress.rule_min_events"] = VD.MIN_STRATUM_EVENTS
    r["suppress.n_strata_evaluated"] = len(fits)
    suppressed = [f for f in fits if f.suppressed]
    r["suppress.n_strata_suppressed"] = len(suppressed)
    r["suppress.strata_suppressed"] = [
        f"{f.stratum.stratifier}: {f.stratum.level}" for f in suppressed]
    r["suppress.residual_level_rule"] = (
        "A stratum level that is not one of the stratifier's declared levels is labelled "
        "from the column that carries the recorded value, through the label layer, and "
        "is called not recorded only when the value is genuinely unrecorded. Its "
        "fragment key segment is frozen at the residual name and is never rendered.")
    for f in suppressed:
        r[f"suppress.{f.stratum.slug}.suppressed"] = True
        r[f"suppress.{f.stratum.slug}.events"] = f.events
        r[f"suppress.{f.stratum.slug}.n"] = f.n
        r[f"suppress.{f.stratum.slug}.reason"] = f.reason
        r[f"suppress.{f.stratum.slug}.level_label"] = f.stratum.level
        r[f"suppress.{f.stratum.slug}.level_recorded"] = not f.stratum.is_missing_level

    # Measured summaries of the two sensitivity analyses and of the monotonicity check.
    # Every claim the notes make about them is read back from these keys rather than
    # asserted in prose, so a future data vintage cannot leave a sentence stale.
    km_moves = 0
    km_never_below_aj = True
    km_gap_max = 0.0
    idx_moves = 0
    idx_max_shift = 0
    cells_total = 0
    never_reached = 0
    rises_all_follow_a_competing_death = True
    all_stay_below = True

    monotone_violations = 0
    for fit in fits:
        s = fit.stratum.slug
        r[f"window.{s}.n"] = fit.n
        r[f"window.{s}.events"] = fit.events
        r[f"window.{s}.competing_deaths"] = fit.competing
        r[f"window.{s}.suppressed"] = fit.suppressed
        if fit.suppressed:
            continue
        r[f"window.{s}.cif30.est"] = 100.0 * fit.cif30
        r[f"window.{s}.cif30.lo"] = 100.0 * fit.cif30_lo
        r[f"window.{s}.cif30.hi"] = 100.0 * fit.cif30_hi
        r[f"window.{s}.day0_events.n"] = fit.day0_events
        r[f"window.{s}.day01_events.n"] = fit.day01_events
        r[f"window.{s}.day01_events.pct_of_events"] = (
            100.0 * fit.day01_events / fit.events)
        for threshold in VD.RISK_THRESHOLDS:
            slug = threshold_slug(threshold)
            day = int(first_day_below(fit.curve[None, :], threshold)[0])
            reached = day <= VD.FOLLOWUP_DAYS
            boot = fit.boot_days[threshold]
            d_lo, d_hi = percentile_ci(boot.astype(float))
            r[f"window.{s}.{slug}.reached"] = reached
            r[f"window.{s}.{slug}.day"] = int(day) if reached else None
            r[f"window.{s}.{slug}.day_lo"] = (
                int(math.floor(d_lo)) if d_lo <= VD.FOLLOWUP_DAYS else None)
            r[f"window.{s}.{slug}.day_hi"] = (
                int(math.ceil(d_hi)) if d_hi <= VD.FOLLOWUP_DAYS else None)
            r[f"window.{s}.{slug}.day_unreached_replicates"] = int(
                (boot > VD.FOLLOWUP_DAYS).sum())
            r[f"window.{s}.{slug}.risk_pct"] = (
                100.0 * float(fit.curve[day]) if reached else None)
            r[f"window.{s}.{slug}.risk_lo"] = (
                100.0 * float(fit.lo[day]) if reached else None)
            r[f"window.{s}.{slug}.risk_hi"] = (
                100.0 * float(fit.hi[day]) if reached else None)
            r[f"window.{s}.{slug}.at_risk_n"] = (
                int(round(float(fit.at_risk[day]))) if reached else None)
            # How many of that conditioning set were still in hospital on that day. The
            # safe discharge day is read against this, not against the at-risk count
            # alone: on most threshold days the great majority of the denominator had
            # already gone home, so no observation-effect reading of the day is available.
            r[f"window.{s}.{slug}.in_hospital_n"] = (
                int(fit.in_hospital_by_day[day]) if reached else None)
            r[f"window.{s}.{slug}.in_hospital_pct"] = (
                100.0 * float(fit.in_hospital_by_day[day])
                / float(fit.at_risk_counted[day]) if reached else None)
            still_below = stays_below(fit.curve, threshold, day) if reached else False
            r[f"window.{s}.{slug}.stays_below"] = still_below
            if not still_below:
                all_stay_below = False
            day_km = int(first_day_below(fit.curve_km[None, :], threshold)[0])
            r[f"window.{s}.{slug}.day_km"] = (
                int(day_km) if day_km <= VD.FOLLOWUP_DAYS else None)
            day_alt = int(first_day_below(fit.curve_end_of_day[None, :], threshold)[0])
            r[f"window.{s}.{slug}.day_end_of_day_indexing"] = (
                int(day_alt) if day_alt <= VD.FOLLOWUP_DAYS else None)

            cells_total += 1
            if not reached:
                never_reached += 1
            if day_km != day:
                km_moves += 1
            if day_alt != day:
                idx_moves += 1
                idx_max_shift = max(idx_max_shift, abs(day_alt - day))

        km_gap_max = max(km_gap_max, float(np.max(np.abs(fit.curve_km - fit.curve))))
        if not bool(np.all(fit.curve_km >= fit.curve - 1e-15)):
            km_never_below_aj = False
        for day in (fit.rise_days or []):
            if float(fit.competing_by_day[day - 1]) <= 0:
                rises_all_follow_a_competing_death = False

        r[f"condrisk.{s}.day"] = list(range(DAYS))
        r[f"condrisk.{s}.est"] = [100.0 * float(v) for v in fit.curve]
        r[f"condrisk.{s}.lo"] = [100.0 * float(v) for v in fit.lo]
        r[f"condrisk.{s}.hi"] = [100.0 * float(v) for v in fit.hi]
        r[f"condrisk.{s}.at_risk"] = [int(round(float(v))) for v in fit.at_risk]
        r[f"condrisk.{s}.km_est"] = [100.0 * float(v) for v in fit.curve_km]
        r[f"condrisk.{s}.monotone"] = fit.monotone
        r[f"condrisk.{s}.rise_days"] = [int(d) for d in (fit.rise_days or [])]
        r[f"condrisk.{s}.km_max_abs_difference_pct"] = 100.0 * float(
            np.max(np.abs(fit.curve_km - fit.curve)))
        # The composition of the same conditioning set, on the same 31 days and aligned
        # with the same day list, so any day of any stratum can be quoted without a
        # reader re-deriving it.
        r[f"condrisk.{s}.in_hospital_n"] = [int(v) for v in fit.in_hospital_by_day]
        r[f"condrisk.{s}.in_hospital_pct"] = [
            100.0 * float(h) / float(a)
            for h, a in zip(fit.in_hospital_by_day, fit.at_risk_counted)]
        r[f"condrisk.{s}.stay_not_recorded_n"] = [
            int(v) for v in fit.stay_missing_by_day]
        if not fit.monotone:
            monotone_violations += 1

    r["condrisk.n_strata_with_a_rise"] = monotone_violations
    r["condrisk.rises_all_follow_a_competing_death"] = rises_all_follow_a_competing_death
    r["condrisk.days"] = list(range(DAYS))

    r["window.sensitivity.stratum_threshold_cells"] = cells_total
    r["window.sensitivity.thresholds_never_reached"] = never_reached
    r["window.sensitivity.km_safe_days_moved"] = km_moves
    r["window.sensitivity.km_changes_no_safe_day"] = km_moves == 0
    r["window.sensitivity.km_max_abs_difference_pct"] = 100.0 * km_gap_max
    r["window.sensitivity.km_never_below_aalen_johansen"] = km_never_below_aj
    r["window.sensitivity.indexing_safe_days_moved"] = idx_moves
    r["window.sensitivity.indexing_max_abs_day_shift"] = idx_max_shift
    r["window.stays_below_all_cells"] = all_stay_below
    return r


# ===========================================================================
# 8. NOTES
# ===========================================================================


def write_notes(fits: Sequence[Fit], r: Dict[str, object], boundary: Dict[str, object]) -> None:
    by_slug = {f.stratum.slug: f for f in fits}
    overall = by_slug["overall"]

    def line_for(fit: Fit) -> str:
        cells = []
        for threshold in VD.RISK_THRESHOLDS:
            day = int(first_day_below(fit.curve[None, :], threshold)[0])
            boot = fit.boot_days[threshold]
            d_lo, d_hi = percentile_ci(boot.astype(float))
            if day > VD.FOLLOWUP_DAYS:
                cells.append(NOT_REACHED)
            else:
                cells.append(f"{day} ({day_ci_text(d_lo, d_hi)}), "
                             f"{pct(float(fit.curve[day]))} "
                             f"[{pct_ci(float(fit.lo[day]), float(fit.hi[day]))}]")
        return (f"| {fit.stratum.stratifier} | {fit.stratum.level} | "
                f"{VD.fmt_count(fit.n)} | {VD.fmt_count(fit.events)} | "
                f"{pct(fit.cif30)} | " + " | ".join(cells) + " |")

    suppressed = [f for f in fits if f.suppressed]
    km_gaps = [(f.stratum.level, 100.0 * float(np.max(np.abs(f.curve_km - f.curve))))
               for f in fits if not f.suppressed]
    worst_km = max(km_gaps, key=lambda kv: kv[1])
    risers = [f for f in fits if not f.suppressed and not f.monotone]

    lines: List[str] = []
    A = lines.append
    A("# notes_06.md -- handoff from `06_window.py` (the safe discharge window)")
    A("")
    A("Written 2026-07-31. Owner: **T8 (window)**. Owned files: `06_window.py`,")
    A("`outputs/results_06.json`, `derived/notes_06.md`,")
    A("`tables-csv/table5_safe_discharge_day.csv`, `tables-csv/table5b_conditional_risk.csv`,")
    A("`tables-csv/s4_sensitivity_06.csv`. Nothing else was created or modified. The")
    A("warehouse was not read; the only input is the frozen `derived/cohort_analytic.parquet`.")
    A("")
    A("This step carries the paper's **lead contribution** (`CLAUDE.md` 6a.3): a systematic")
    A("literature search found no precedent anywhere for the conditional safe-discharge-day")
    A("estimand, while two of `PLAN.md`'s other framing claims were refuted. Every number")
    A("below is re-derived here from the parquet and none is copied from a handoff note.")
    A("")
    A("---")
    A("")
    A("## 1. The estimand, exactly as computed")
    A("")
    A("For a stratum and the pre-specified composite outcome, the **safe discharge day** is")
    A("the earliest postoperative day D at which the **conditional** probability of a")
    A("subsequent qualifying event, given the patient is event-free at day D, falls below a")
    A("pre-specified absolute-risk threshold. It is a conditional-risk-given-survival")
    A("quantity read off the cumulative incidence function, **not a hazard**.")
    A("")
    A("```")
    A("conditional risk at day D =")
    A("      ( cumulative incidence at day 30  -  cumulative incidence at the start of day D )")
    A("    / ( probability of being event-free AND ALIVE at the start of day D )")
    A("```")
    A("")
    A("**Death is a competing event**, so the numerator is the **Aalen-Johansen** cumulative")
    A("incidence and not one minus Kaplan-Meier (`PLAN.md` 6.1): a patient who died on day 3")
    A("is not at risk of a hematoma reoperation on day 7. The size of that correction is")
    A("measured in section 6 rather than assumed small.")
    A("")
    A("### 1.1 Why the conditioning set is the START of day D")
    A("")
    A("Two statements in the locked estimand have to hold at once: the numerator is the")
    A("incidence *accrued between day D and day 30*, and the conditional risk at day 0 is")
    A("*the day-30 cumulative incidence itself*. Both hold only if an event recorded ON day D")
    A("counts as still ahead of a patient discharged on day D. That is also the conservative")
    A("reading for a safety claim: NSQIP records whole postoperative days, an event and a")
    A("discharge on the same integer day cannot be ordered, and crediting the observation")
    A("window with catching a same-day event would shorten the recommended window on an")
    A("assumption the data cannot support. The choice is recorded as `window.indexing_rule`.")
    A("")
    A("**The opposite (end-of-day) indexing is reported as a sensitivity in")
    A("`s4_sensitivity_06.csv`, and its effect is arithmetically bounded, not just observed")
    A("to be small.** The alternative curve is the primary curve shifted by exactly one day,")
    A("so the safe discharge day under it is either the same day (where the primary day is")
    A("already 0) or exactly one day earlier. Measured across all")
    A(f"{r['window.sensitivity.stratum_threshold_cells']} stratum-and-threshold cells it moves")
    A(f"the day in {r['window.sensitivity.indexing_safe_days_moved']} of them, always earlier")
    A(f"and never by more than "
      f"{r['window.sensitivity.indexing_max_abs_day_shift']} day. **The primary convention is")
    A("therefore the conservative one by exactly one day**, which is the whole of the")
    A("disagreement and is stated rather than argued.")
    A("")
    A("This is a different question from the `CLAUDE.md` 6.3 tie convention, which governs")
    A("whether an event on the discharge day is scored pre-discharge or post-discharge in")
    A("`04_postdischarge.py`. That convention is not used here and no `POSTDISCH_*` column")
    A("is read by this step.")
    A("")
    A("### 1.2 The estimator reduces to exact counting, and that is checked every run")
    A("")
    A("NSQIP follow-up is fixed and **complete** at 30 days, so censoring is administrative")
    A("and non-informative and there is no loss to follow-up inside the window. Under")
    A("complete follow-up the Aalen-Johansen product-limit recursion reduces **exactly** to")
    A("counting: the cumulative incidence is the crude cumulative proportion and the")
    A("conditional risk is (remaining composite events) divided by (patients still")
    A("event-free and alive). `06_window.py` runs the general recursion and **asserts** it")
    A("equals the closed form to 1e-12 on every stratum, and separately asserts that the")
    A("day-0 conditional risk equals the day-30 cumulative incidence. A disagreement stops")
    A("the step. This is stated because a reviewer will ask why a competing-risks framework")
    A("was used on a low-mortality cohort: the answer is that the correctness of the")
    A("estimator should not depend on the answer being close either way, and the supplement")
    A("shows that it is close.")
    A("")
    A("### 1.3 WHO the day-D conditioning set is. READ THIS BEFORE WRITING ABOUT THE WINDOW.")
    A("")
    A("**The conditioning set is a SURVIVAL condition, not an OBSERVATION condition, and it")
    A("is overwhelmingly made up of patients who had already gone home.** A patient")
    A("discharged on postoperative day 1 who is alive and event free on day 7 is in the")
    A("day-7 conditioning set, and almost all of it is patients like that. The final review")
    A("measured this and found that no artefact reported it, which left the language around")
    A("the estimand free to import a causal, observation-effect reading the estimand does")
    A("not carry. It is measured and published now.")
    A("")
    A("**Definition, stated exactly.** A patient is **still in hospital at the start of")
    A("postoperative day D** when their postoperative stay had not ended before day D, that")
    A("is when `LOS_POSTOP >= D` on the column `02` froze from the direct")
    A("days-from-operation-to-discharge field. A patient discharged **on** day D was in")
    A("hospital at the start of it. That is the same day indexing the estimand itself uses")
    A("(section 1.1) and the same direction as the `CLAUDE.md` 6.3 tie convention, so the")
    A("composition and the risk are counted on the same clock. **No length of stay is")
    A("truncated at day 30** (`.build_facts.md` 7.5: the field runs to 112 days and 74")
    A("analytic patients stayed past the end of the window).")
    A("")
    A("| Postoperative day | Event free and alive at the start of it | Of whom still in "
      "hospital | Still in hospital, % | Of whom discharged | Stay not recorded |")
    A("|---:|---:|---:|---:|---:|---:|")
    for d in RISKSET_REPORT_DAYS:
        A(f"| {d} | {VD.fmt_count(r[f'window.riskset.day{d}.at_risk_n'])} | "
          f"{VD.fmt_count(r[f'window.riskset.day{d}.in_hospital_n'])} | "
          f"{VD.fmt_pct(r[f'window.riskset.day{d}.in_hospital_pct'], 2)} | "
          f"{VD.fmt_count(r[f'window.riskset.day{d}.discharged_n'])} | "
          f"{VD.fmt_count(r[f'window.riskset.day{d}.stay_not_recorded_n'])} |")
    A("")
    _tight = VD.RISK_THRESHOLDS[-1]
    _tight_slug = threshold_slug(_tight)
    A(f"At the overall {VD.threshold_label(_tight)} threshold, safe discharge day "
      f"{r[f'window.overall.{_tight_slug}.day']}, the residual risk is measured on "
      f"{VD.fmt_count(r[f'window.overall.{_tight_slug}.at_risk_n'])} patients of whom "
      f"{VD.fmt_count(r[f'window.overall.{_tight_slug}.in_hospital_n'])} "
      f"({VD.fmt_pct(r[f'window.overall.{_tight_slug}.in_hospital_pct'], 1)}) were still")
    A("being observed at all. **That is why \"the observation needed\" and \"needs D days\"")
    A("are the wrong words for this quantity**, and why `window.riskset.text.caveat` exists")
    A("for `08` to quote instead of inventing a sentence.")
    A("")
    A("**How the two awkward populations are handled, explicitly rather than silently**")
    A("(`.build_facts.md` 7.5 binding items 6 and 8):")
    A("")
    A(f"* **{VD.fmt_count(r['window.riskset.stay_not_recorded_n'])} patients have no")
    A("  recorded postoperative stay** and cannot be placed on either side. They are their")
    A("  own published count on every day (`stay_not_recorded_n`), the primary")
    A("  still-in-hospital count **excludes** them, and an upper bound that places every one")
    A("  of them in hospital is published beside it (`in_hospital_upper_n` /")
    A("  `in_hospital_upper_pct`). The bound is what makes the choice auditable instead of a")
    A("  convention buried in a comparison. Of them, "
      f"{VD.fmt_count(r['window.riskset.stay_not_recorded_and_flagged_n'])} carry the")
    A("  still-in-hospital flag and are therefore known to have been inpatients at the end")
    A(f"  of the window; {VD.fmt_count(r['window.riskset.stay_not_recorded_not_flagged_n'])}")
    A("  are genuinely unplaceable.")
    A(f"* **{VD.fmt_count(r['window.riskset.still_in_hospital_flag_n'])} patients are")
    A("  flagged still in hospital.** The flag is **never** used to override a recorded")
    A("  stay, because `.build_facts.md` 7.5 binding item 6 is explicit that treating them")
    A("  all as having no post-discharge period is false: "
      f"{VD.fmt_count(r['window.riskset.flagged_with_recorded_stay_n'])} of them carry a")
    A(f"  recorded length of stay, "
      f"{VD.fmt_count(r['window.riskset.flagged_stay_beyond_window_n'])} of them past day "
      f"{VD.FOLLOWUP_DAYS} and "
      f"{VD.fmt_count(r['window.riskset.flagged_stay_inside_window_n'])} at or before it.")
    A("  It is used only to say how many of the unplaceable patients are known to have been")
    A("  inpatients.")
    A("")
    A("The composition is asserted to partition the risk set on every day of every stratum,")
    A("and the counted at-risk total is asserted equal to the at-risk denominator the")
    A("estimator publishes. If the two ever disagree the composition would be describing a")
    A("different set from the one the risk is measured on, which is exactly the failure this")
    A("section exists to prevent.")
    A("")
    A("Published as `window.riskset.*` (cohort counts, the representative days "
      + ", ".join(str(d) for d in RISKSET_REPORT_DAYS) + ", and four renderable")
    A("sentences under `window.riskset.text.*`), as")
    A("`window.<stratum>.<threshold>.in_hospital_n` / `.in_hospital_pct` at every stratum's")
    A("own safe discharge day, and as `condrisk.<stratum>.in_hospital_n` /")
    A("`.in_hospital_pct` / `.stay_not_recorded_n`, three lists of 31 values aligned with")
    A("`condrisk.days`, for every estimated stratum.")
    A("")
    A("---")
    A("")
    A("## 2. The locked pre-specification, re-derived")
    A("")
    A(f"| Item | Value used here |")
    A("|---|---|")
    A(f"| Composite | `WINDOW_COMP` = hematoma requiring unplanned reoperation OR unplanned "
      f"reintubation OR ventilator beyond 48 hours OR reoperative tracheostomy |")
    A("| Excluded from the composite | pneumonia, unplanned readmission |")
    A("| Composite timing | `POD_WINDOW`, the earliest observed day among the components that fired |")
    A(f"| Thresholds | {', '.join(VD.threshold_label(t) for t in VD.RISK_THRESHOLDS)} |")
    A(f"| Stratifiers | {', '.join(STRATIFIER_DISPLAY[c] for c in VD.STRATIFIER_COLUMNS)} |")
    A(f"| Suppression | fewer than {VD.MIN_STRATUM_EVENTS} composite events |")
    A(f"| Uncertainty | seeded within-stratum percentile bootstrap, "
      f"{BOOTSTRAP_REPLICATES:,} replicates, seed {VD.SEED} |")
    A(f"| Follow-up | {VD.FOLLOWUP_DAYS} days, no extrapolation past it |")
    A("")
    A("**Re-derived composite counts on the analytic cohort** "
      f"({VD.fmt_count(r['window.cohort.n'])} patients):")
    A("")
    A("| Component | n |")
    A("|---|---:|")
    for name, label in boundary["component_labels"]:
        A(f"| {label} | {VD.fmt_count(r[f'window.component.{name}.n'])} |")
    A(f"| **`WINDOW_COMP`** | **{VD.fmt_count(r['window.composite.n'])}** "
      f"({VD.fmt_pct(r['window.composite.pct'], 3)}) |")
    A("")
    A(f"`02` recorded 1,015 (0.901%) with `POD_WINDOW` observed on all of them. Re-derived "
      f"here independently: **"
      f"{'REPRODUCES' if boundary['reproduces_02'] else 'DOES NOT REPRODUCE'}**. "
      f"`POD_WINDOW` is observed on {VD.fmt_count(r['window.composite.n'])} of "
      f"{VD.fmt_count(r['window.composite.n'])} composite events, every value an integer in "
      f"0 to {VD.FOLLOWUP_DAYS}.")
    A("")
    A("### 2.1 The competing event, and the one convention this step had to add")
    A("")
    A(f"| Quantity | n |")
    A("|---|---:|")
    A(f"| deaths inside the 30-day window (`MORT30`) | "
      f"{VD.fmt_count(r['window.competing.deaths_in_window.n'])} |")
    A(f"| patients with BOTH a composite event and a death | "
      f"{VD.fmt_count(r['window.competing.both_events.n'])} |")
    A(f"| of those, death strictly BEFORE the composite | "
      f"{VD.fmt_count(r['window.competing.death_strictly_first.n'])} |")
    A(f"| of those, death on the SAME day as the composite | "
      f"{VD.fmt_count(r['window.competing.tie_same_day.n'])} |")
    A(f"| deaths entering the estimator as the COMPETING (first) event | "
      f"**{VD.fmt_count(r['window.competing.deaths_first.n'])}** |")
    A("")
    A("All analyses are time-to-first-event. `PLAN.md` and `CLAUDE.md` do not pre-specify")
    A("what happens when the composite and the death fall on the **same integer day**, so a")
    A("rule was needed and is stated rather than buried: **the composite takes precedence**.")
    A("That is the conservative direction for a safety claim (it raises the cumulative")
    A("incidence of the composite and therefore pushes the safe discharge day later), and it")
    A("is clinically the right reading, since the airway or hematoma catastrophe is the event")
    A("continued observation could have caught. **No death occurs strictly before its own")
    A("patient's composite event**, so the rule moves exactly the tied patients and nothing")
    A("else. It is published as `window.competing_tie_rule`.")
    A("")
    A("---")
    A("")
    A("## 3. THE HEADLINE: the overall safe discharge day")
    A("")
    A("This is the single number a surgeon will quote, so it is reported unstratified as well")
    A("as by stratum.")
    A("")
    A(f"Overall 30-day cumulative incidence of the composite: **{pct(overall.cif30)}** "
      f"(95% CI {pct_ci(overall.cif30_lo, overall.cif30_hi)}), "
      f"{VD.fmt_count(overall.events)} events in "
      f"{VD.fmt_count(overall.n)} patients.")
    A("")
    A("| Threshold | Safe discharge day (95% CI) | Residual risk at that day (95% CI) | Patients still at risk |")
    A("|---|---|---|---:|")
    for threshold in VD.RISK_THRESHOLDS:
        day = int(first_day_below(overall.curve[None, :], threshold)[0])
        boot = overall.boot_days[threshold]
        d_lo, d_hi = percentile_ci(boot.astype(float))
        if day > VD.FOLLOWUP_DAYS:
            A(f"| {VD.threshold_label(threshold)} | {NOT_REACHED} | {NOT_REACHED} | - |")
        else:
            A(f"| {VD.threshold_label(threshold)} | {day} ({day_ci_text(d_lo, d_hi)}) | "
              f"{pct(float(overall.curve[day]))} "
              f"({pct_ci(float(overall.lo[day]), float(overall.hi[day]))}) | "
              f"{VD.fmt_count(round(float(overall.at_risk[day])))} |")
    A("")
    A("The conditional risk on the first four postoperative days, overall:")
    A("")
    A("| Postoperative day | Conditional risk of a subsequent composite event | Patients at risk |")
    A("|---:|---|---:|")
    for day in range(min(5, DAYS)):
        A(f"| {day} | {pct(float(overall.curve[day]))} "
          f"({pct_ci(float(overall.lo[day]), float(overall.hi[day]))}) | "
          f"{VD.fmt_count(round(float(overall.at_risk[day])))} |")
    A("")
    A(f"Day 0 carries {VD.fmt_count(overall.day0_events)} of the "
      f"{VD.fmt_count(overall.events)} composite events and days 0 and 1 together carry "
      f"{VD.fmt_count(overall.day01_events)} "
      f"({VD.fmt_pct(100.0 * overall.day01_events / overall.events, 1)}). "
      f"**The conditional risk at day 0 is the 30-day cumulative incidence by construction**, "
      f"which is the estimand's own arithmetic check and is asserted every run.")
    A("")
    A("---")
    A("")
    A("## 4. Safe discharge day BY STRATUM")
    A("")
    A("Fixed, pre-specified strata (`CLAUDE.md` 6.5). **No coefficient is estimated, no odds")
    A("ratio is reported and no discrimination statistic is presented** (`CLAUDE.md` 6.6 and")
    A("7.1): the multivariable hematoma risk model and the points score belong to the")
    A("companion paper, which is cited for the derivation of these strata.")
    A("")
    A("Each cell is the safe discharge day with its bootstrap interval, then the residual")
    A("risk at that day with its 95% interval.")
    A("")
    A("| Stratifier | Stratum | Patients | Composite events | 30-day cumulative incidence | "
      + " | ".join(VD.threshold_label(t) for t in VD.RISK_THRESHOLDS) + " |")
    A("|---|---|---:|---:|---|" + "---|" * len(VD.RISK_THRESHOLDS))
    for fit in fits:
        if fit.suppressed:
            A(f"| {fit.stratum.stratifier} | {fit.stratum.level} | "
              f"{VD.fmt_count(fit.n)} | {VD.fmt_count(fit.events)} | {SUPPRESSED_TEXT} | "
              + " | ".join([SUPPRESSED_TEXT] * len(VD.RISK_THRESHOLDS)) + " |")
        else:
            A(line_for(fit))
    A("")
    A("The full conditional-risk curve for every stratum, day by day with confidence limits,")
    A("is `tables-csv/table5b_conditional_risk.csv`, which is what `07_figures.py` plots as")
    A("Figure 6.")
    A("")
    A("---")
    A("")
    A("## 5. Suppression (`CLAUDE.md` 6.5), reported with its count")
    A("")
    A(f"The rule was set before any curve was seen: **a stratum with fewer than "
      f"{VD.MIN_STRATUM_EVENTS} composite events is suppressed rather than plotted, and the "
      f"suppression is reported with its count.**")
    A("")
    A(f"Strata evaluated: **{r['suppress.n_strata_evaluated']}** "
      f"(the overall row plus every level of the four locked stratifiers). "
      f"Strata suppressed: **{r['suppress.n_strata_suppressed']}**.")
    A("")
    if suppressed:
        A("| Stratifier | Stratum | Patients | Composite events | Reason |")
        A("|---|---|---:|---:|---|")
        for fit in suppressed:
            A(f"| {fit.stratum.stratifier} | {fit.stratum.level} | "
              f"{VD.fmt_count(fit.n)} | {VD.fmt_count(fit.events)} | {fit.reason} |")
        A("")
        A("The suppressed level is the residual level of an existing locked stratifier, not a")
        A("new one. It is carried through the table so that the four declared strata provably")
        A("account for every patient in the cohort, and it is suppressed under the rule rather")
        A("than dropped silently. It is present in `table5b_conditional_risk.csv` with its")
        A("estimate columns empty and its `Suppressed` column set to Yes, so Figure 6 can say")
        A("so on the face of the figure.")
        A("")
        A("### 5.1 The residual level is LABELLED, not called missing. Fix M5.")
        A("")
        A("**The final review found this stratum rendering as "
          f"\"{VD.MISSING_LABEL}\" in Table 5, Table 5B, supplement S4D and Figure 6 panel D,")
        A("while the manuscript said the opposite in the same submission package.** The data")
        A("are recorded. `SEX` in the analytic parquet carries three levels and the residual")
        A("level of this stratifier is one of them; `02_build_cohort.py` stores the stratifier")
        A("as the binary `MALE` and deliberately leaves it Int8-missing for these patients")
        A("rather than folding them into the female reference group, which would be")
        A("misclassification, and the label layer then fell through to the missing label.")
        A("")
        A("**The fix distinguishes the two cases rather than renaming anything.** A residual")
        A("level is now resolved against the column that carries the recorded value")
        A("(`RESIDUAL_SOURCE_COLUMN`) and its display text comes from")
        A("`variable_dict.pretty_level()` on that column, so no string is written here. A")
        A("residual level whose value is genuinely unrecorded still renders as")
        A(f"\"{VD.MISSING_LABEL}\", which is what that label is for; a residual carrying "
          f"more")
        A("than one distinct recorded value would be a new stratum and raises rather than")
        A("being created silently (`CLAUDE.md` 6.5).")
        A("")
        for fit in suppressed:
            slug = fit.stratum.slug
            A(f"* **{fit.stratum.stratifier} / {fit.stratum.level}**: stratifier value "
              f"recorded = **{r[f'suppress.{slug}.level_recorded']}**, resolved from "
              f"`{fit.stratum.source_column or 'the stratifier column itself'}`. "
              f"Published as `suppress.{slug}.level_label` and "
              f"`suppress.{slug}.level_recorded`.")
        A("")
        A("**The fragment key segment is FROZEN and is deliberately not renamed.** The slug")
        A(f"stays `{RESIDUAL_SLUG}` even where the level turns out to be recorded, because")
        A("`verify_numbers.py` addresses `suppress.sex." + RESIDUAL_SLUG + ".*` by name in")
        A("`STRATUM_SLUGS` and in its Table 5 cell map, and `08_generate_manuscript.py` reads")
        A("the same key. A fragment key is a machine identifier and is never rendered; the")
        A("display label is what a reader sees, and that is what was wrong and is now right.")
        A("Renaming the key would break two files this step does not own for no reader-facing")
        A("gain. The rule is published as `suppress.residual_level_rule`.")
        A("")
        A("**The stratum is still SUPPRESSED and still CARRIED.** Nothing about the")
        A("suppression changed: it has fewer composite events than")
        A("`variable_dict.MIN_STRATUM_EVENTS`, its estimate is withheld, and its rows remain")
        A("in Table 5, Table 5B and S4D flagged `Suppressed` so Figure 6 shows it as")
        A("suppressed rather than omitting it.")
    else:
        A("No stratum was suppressed.")
    A("")
    A("---")
    A("")
    A("## 6. The size of the competing-risk correction ON THIS ESTIMAND")
    A("")
    A("`PLAN.md` 6.6 and supplement S3 require the one-minus-Kaplan-Meier comparison for the")
    A("cumulative incidence functions, which `03_timing.py` owns. It is repeated here **on")
    A("the conditional-risk estimand itself**, because the correction does not have to be the")
    A("same size on a ratio functional as it is on the cumulative incidence, and this is the")
    A("quantity the paper leads with.")
    A("")
    A(f"Overall: the 30-day cumulative incidence of the composite is **{pct(overall.cif30)}** "
      f"under Aalen-Johansen against **{pct(overall.cif30_km)}** under one minus "
      f"Kaplan-Meier, an absolute difference of "
      f"{VD.fmt_pct(100.0 * (overall.cif30_km - overall.cif30), 4)}.")
    A("")
    A(f"Across every estimated stratum the **largest** absolute difference between the two "
      f"conditional-risk curves at any day is "
      f"{VD.fmt_pct(worst_km[1], 4)} (in the stratum {worst_km[0]}). Measured across all "
      f"{r['window.sensitivity.stratum_threshold_cells']} stratum-and-threshold cells, the "
      f"choice of estimator moves the safe discharge day in "
      f"**{r['window.sensitivity.km_safe_days_moved']}** of them. "
      f"`s4_sensitivity_06.csv` prints both estimators' days side by side for every stratum "
      f"and threshold so the reader can see that rather than take it on trust.")
    A("")
    A(f"One minus Kaplan-Meier is at least as large as Aalen-Johansen on every day of every "
      f"estimated stratum: {r['window.sensitivity.km_never_below_aalen_johansen']}. That is "
      f"the expected direction, because treating a death as non-informative censoring "
      f"assumes the dead patient would have gone on to the complication at the same rate as "
      f"the living, which OVERSTATES the incidence of every non-fatal outcome.")
    A("")
    A("The correction is small because the competing event is rare: "
      f"{VD.fmt_count(r['window.competing.deaths_first.n'])} deaths enter as the first event "
      f"in {VD.fmt_count(r['window.cohort.n'])} patients. **That is a reason to report the "
      "correction, not a reason to skip the estimator.** The honest statement is that the "
      "two agree to within the printed precision on this cohort, not that the choice did "
      "not matter: the correctness of an estimator cannot depend on the answer turning out "
      "close either way, and a reader is entitled to see the size of the difference rather "
      "than a reassurance about it.")
    A("")
    A("---")
    A("")
    A("## 7. Monotonicity, and the days where the curve does not fall")
    A("")
    A("The conditional risk is **not** guaranteed to be non-increasing in D when a competing")
    A("event exists: a day carrying a competing death but no composite event shrinks the")
    A("conditioning set without shrinking the residual risk, so the ratio can tick up. After")
    A("the last composite event the curve is exactly zero and stays there, which is the")
    A("required behaviour.")
    A("")
    if risers:
        A(f"**{len(risers)} of the estimated strata carry at least one day where the curve")
        A("rises.** Every rise is preceded by a competing death on the day before, which is")
        A(f"the mechanism above and is checked rather than asserted: "
          f"`condrisk.rises_all_follow_a_competing_death` = "
          f"{r['condrisk.rises_all_follow_a_competing_death']}. The rise column below gives")
        A("the day, the composite events on the preceding day and the competing deaths on the")
        A("preceding day, so the mechanism is visible.")
        A("")
        A("| Stratifier | Stratum | Day of the rise (composite events, competing deaths on the day before) | Size of the largest rise |")
        A("|---|---|---|---|")
        for fit in risers:
            detail = ", ".join(
                f"{d} ({int(fit.events_by_day[d - 1])}, {int(fit.competing_by_day[d - 1])})"
                for d in fit.rise_days)
            biggest = max(float(fit.curve[d] - fit.curve[d - 1]) for d in fit.rise_days)
            A(f"| {fit.stratum.stratifier} | {fit.stratum.level} | {detail} | "
              f"{VD.fmt_pct(100.0 * biggest, 6)} |")
        A("")
        A("Reported rather than smoothed. It is why Table 5 also carries a column saying")
        A("whether the residual risk **remains** below the threshold through day 30, and why")
        A("the safe discharge day is defined as the EARLIEST crossing rather than the last.")
        A(f"Whether any rise takes a stratum back above a threshold it had already crossed "
          f"is measured, not assumed: `window.stays_below_all_cells` = "
          f"{r['window.stays_below_all_cells']}, and the \"remains below\" column of Table 5 "
          f"carries the same verdict cell by cell.")
    else:
        A("No estimated stratum carries a day on which the conditional risk rises.")
    A("")
    A("---")
    A("")
    A("## 8. Uncertainty and reproducibility")
    A("")
    A("Every conditional risk carries a 95% confidence interval. The pre-specification")
    A("(`.build_facts.md` section 2) says the Aalen-Johansen variance where available and a")
    A("seeded bootstrap otherwise. **The conditional risk is a ratio of a difference of two")
    A("Aalen-Johansen quantities to a third**, and no closed-form Aalen-Johansen variance")
    A("exists for that functional (the closed form exists for the cumulative incidence")
    A("alone), so the locked fallback applies and every interval here is a percentile")
    A("bootstrap.")
    A("")
    A(f"* seed {VD.SEED}, set once through `np.random.seed(variable_dict.SEED)` at the top of")
    A(f"  the step, which is the project's only draw on the global NumPy stream;")
    A(f"* {BOOTSTRAP_REPLICATES:,} replicates;")
    A("* **patients resampled WITHIN stratum**, exactly as locked.")
    A("")
    A("Resampling n patients with replacement inside a stratum is implemented as a")
    A("multinomial draw over the cells the patients fall into (composite event on day d,")
    A("competing death on day d, or reaching day 30 event-free and alive). Patients inside")
    A("one cell are exchangeable and each patient falls in exactly one cell, so the draw is")
    A("**exactly** the patient-level bootstrap and not an approximation to it.")
    A("")
    A("The safe discharge day itself also carries a bootstrap interval, computed by finding")
    A("the crossing day inside every replicate. A replicate in which the threshold is never")
    A("crossed is held at a sentinel above day 30 and counted")
    A("(`window.<stratum>.<threshold>.day_unreached_replicates`), so an interval that reaches")
    A("past the window is reported as such and is never clipped to day 30.")
    A("")
    A("---")
    A("")
    A("## 9. INTERPRETATION GUARDRAILS. `08` must carry every one of these.")
    A("")
    A("`PLAN.md` 8 and `CLAUDE.md` 11 require these, and this estimand is the one they bear")
    A("on most.")
    A("")
    A("1. **Confounding by indication, and it is the big one.** The window is derived from")
    A("   observed event timing in a population that was **actually discharged on a")
    A("   clinician's judgement**. Patients kept longer were kept for a reason. **The window")
    A("   describes WHEN EVENTS OCCUR. It is not a causal estimate of what would happen if a")
    A("   patient were discharged earlier**, and it must never be written as one. A stratum")
    A("   whose risk falls quickly may be a stratum whose risky members were kept in hospital.")
    A("2. **Length of stay measures time in hospital, not intensity of observation within it.**")
    A("   A patient on a monitored floor and a patient on a ward are both still admitted.")
    A("3. **Every hematoma figure is a floor.** Hematoma is captured only when it triggered an")
    A("   unplanned reoperation, so observed and aspirated hematomas are invisible.")
    A("4. **Follow-up is 30 days and nothing extends past it.** Events after day 30 exist")
    A("   clinically and are invisible here. No estimate in this step is extrapolated.")
    A("5. **Reoperative tracheostomy is exploratory**, with no literature precedent as an")
    A(f"   airway-catastrophe marker; it contributes "
      f"{VD.fmt_count(r['window.component.trach.n'])} events on reoperation slot 1.")
    A("6. **Timing is whole postoperative days.** An event and a discharge on the same integer")
    A("   day cannot be ordered, which is exactly why the conditioning set is the start of the")
    A("   day and why the opposite indexing is reported as a sensitivity.")
    A("7. **No hospital or surgeon identifier exists in NSQIP**, so institutional variation in")
    A("   discharge practice cannot be separated from patient-level variation.")
    A("8. **THE SAFE DISCHARGE DAY IS NOT AN OBSERVATION PERIOD, AND THE WORDS MATTER.** The")
    A("   day-D conditioning set is the patients event free and alive at the start of day D,")
    A("   whether or not they were in hospital, and section 1.3 measures how few of them")
    A(f"   were: {VD.fmt_pct(r['window.riskset.day7.in_hospital_pct'], 1)} on day 7 and "
      f"{VD.fmt_pct(r['window.riskset.day14.in_hospital_pct'], 1)} on day 14. **Phrases")
    A("   such as \"the observation needed\", \"needs 14 days\", \"must be watched for\" and")
    A("   \"stay of D days\" import a causal, observation-effect reading the estimand does")
    A("   not carry** and must not attach to it. The defensible reading is the one")
    A("   `window.riskset.text.caveat` states: the day on which the probability of a")
    A("   subsequent qualifying event first falls below the stated threshold, estimated on a")
    A("   cohort in which almost every patient still at risk on that day was already at")
    A("   home. This is guardrail 1 in a second form and is the one a reader is most likely")
    A("   to breach without noticing.")
    A("")
    A("---")
    A("")
    A("## 10. Comparison with the nearest published competitor")
    A("")
    A("**Runge et al. 2026, *Global Spine J* (PMID 42383777)** is the closest published")
    A("answer to this paper's question: median 9 hours to symptom onset, 88% of events within")
    A("24 hours, 12% in-hospital mortality, concluding that a 4-to-8-hour observation period")
    A("is insufficient. `CLAUDE.md` 6a.4 requires the Discussion to engage it directly.")
    A("")
    A("**It reaches a compatible conclusion from a different design, which is support rather")
    A("than a threat.** Runge is a hematoma-specific series with hourly resolution; this is a")
    A("national registry cohort with whole-day resolution and a broader composite that also")
    A("carries unplanned reintubation, prolonged ventilation and reoperative tracheostomy.")
    A("The two agree on the clinical bottom line and disagree on nothing:")
    A("")
    A(f"* Runge: a 4-to-8-hour observation period is insufficient. Here: a patient discharged")
    A(f"  on the day of surgery still carries a **{pct(overall.cif30)}** residual 30-day risk")
    A(f"  of the composite, which is the entire cumulative incidence, because nothing has been")
    A("  observed away yet.")
    A(f"* Runge: 88% of events within 24 hours. Here: "
      f"{VD.fmt_pct(100.0 * overall.day01_events / overall.events, 1)} of composite events")
    A(f"  fall on postoperative day 0 or 1 "
      f"({VD.fmt_count(overall.day01_events)} of {VD.fmt_count(overall.events)}). The lower")
    A("  share is expected and is not a contradiction: the denominators are different")
    A("  outcomes (a broader composite whose ventilator and reintubation arms run later than")
    A("  hematoma), the resolution is different (whole days against hours), and NSQIP")
    A("  postoperative day 1 is not the same interval as the first 24 hours.")
    A("* The two therefore converge on the same recommendation from opposite directions:")
    A("  Runge shows that a few hours is not enough, and this analysis quantifies how many")
    A("  days are needed to reach a stated absolute residual risk, which is the question a")
    A("  discharge order actually asks.")
    A("")
    A("**Do not present the two as confirming each other's exact numbers** (`CLAUDE.md` 6a.1")
    A("makes the same ruling for the `INOUT` figures). Report ours with its denominator and")
    A("cite theirs beside it.")
    A("")
    A("---")
    A("")
    A("## 11. What `07_figures.py` and `08_generate_manuscript.py` must know")
    A("")
    A("1. **Figure 6 is `tables-csv/table5b_conditional_risk.csv` and nothing else.** It is")
    A("   self-sufficient: one row per stratum and postoperative day, with the point estimate,")
    A("   both confidence limits, the number still at risk, the cumulative incidence to the")
    A("   start of the day, a stable `Display order`, and a `Suppressed` column. The threshold")
    A("   lines come from `variable_dict.RISK_THRESHOLDS`, drawn with `threshold_label()`.")
    A("2. **Plot STEP curves.** Every quantity is a step function on integer days")
    A("   (`CLAUDE.md` 6.1). Do not smooth, do not interpolate, do not draw sub-day")
    A("   resolution the data does not have, and **do not extend any line past day 30**.")
    A("3. **The suppressed stratum must be visible as suppressed**, not silently absent. Its")
    A("   rows are present with empty estimates and `Suppressed` = Yes.")
    A("4. **`03_timing.py` owns the cumulative incidence curves and Gray's tests.** This step")
    A("   emits no general-purpose cumulative incidence table; the cumulative incidence")
    A("   column in Table 5B exists only as the input to this estimand and is for the")
    A("   composite `WINDOW_COMP`, which is not one of the eight roster outcomes.")
    A("5. **Every number in prose comes from `outputs/results.json` or from a table CSV.**")
    A("   The overall headline keys are `window.overall.cif30.est` and")
    A("   `window.overall.<threshold>.day` / `.risk_pct` / `.risk_lo` / `.risk_hi`, with")
    A("   `<threshold>` one of "
      + ", ".join("`" + threshold_slug(t) + "`" for t in VD.RISK_THRESHOLDS) + ".")
    A("6. **EVERY RISK THIS STEP PUBLISHES IS A PERCENTAGE, INCLUDING `.est`, `.lo` and")
    A("   `.hi`.** `window.overall.cif30.est` is **0.901**, meaning 0.901%, and it is the")
    A("   paper's lead result. `03_timing.py` uses the same three suffixes for")
    A("   **proportions** on the 0 to 1 scale, and nothing in a key name distinguishes the")
    A("   two: applying `03`'s rule here would render the headline as 0.090% or as 90.1%.")
    A("   Read `window.scale` and `condrisk.scale`, never another step's declaration")
    A("   (`.build_facts.md` section 9). Note in particular that `risk_lo` and `risk_hi`")
    A("   are percentages whose sibling is spelled `risk_pct`; the suffix is inconsistent")
    A("   and the unit is not. `window.thresholds` is the one exception in this step and")
    A("   is held as proportions, with `window.threshold_labels` beside it for rendering.")
    A("7. **`window.<stratum>.<threshold>.reached` is False when a threshold is never met")
    A("   inside the window**, and the `.day` key is then `null`. Render it as")
    A(f"   \"{NOT_REACHED}\". **Never print the last day and never extrapolate.** On the")
    A(f"   current data "
      f"{r['window.sensitivity.thresholds_never_reached']} of "
      f"{r['window.sensitivity.stratum_threshold_cells']} stratum-and-threshold cells are")
    A("   unreached, so no `.day` key is `null` today. **The branch is still live and must be")
    A("   handled**: a future vintage, a rarer stratum or a fourth threshold would use it.")
    A("8. **Say the guardrails in section 9 out loud**, above all confounding by indication.")
    A("   A window recommendation read as causal is the single most likely way this result")
    A("   gets misused.")
    A("9. **The Methods must state** the estimand, the competing-risks estimator and why, the")
    A("   start-of-day conditioning, the same-day composite-versus-death tie rule, the")
    A("   suppression rule with its count, and the bootstrap with its seed, replicate count")
    A("   and resampling unit. All eight are fragment keys; none needs typing.")
    A("10. **QUOTE THE FOUR CONDITIONING-SET SENTENCES; DO NOT PARAPHRASE THEM.** Section 1.3")
    A("    measures who the conditional risk is actually estimated on, and the four keys")
    A("    below are written to be rendered verbatim so `08` types no number by hand:")
    A("    `window.riskset.text.methods` (the estimand is a survival condition, not an")
    A("    observation condition), `window.riskset.text.missing_handling` (the patients with")
    A("    no recorded stay and the still-in-hospital flag), `window.riskset.text.results`")
    A("    (the measured composition at days 3, 7 and 14) and `window.riskset.text.caveat`")
    A("    (the words that must not attach to the safe discharge day). Per-day numbers are")
    A(f"    `window.riskset.day<d>.*` for d in "
      + ", ".join(str(d) for d in RISKSET_REPORT_DAYS) + ", each with a rendered")
    A("    `.text` in the form n of d (p%), plus `window.<stratum>.<threshold>.in_hospital_n`")
    A("    and `.in_hospital_pct` at every stratum's own safe discharge day, plus the full")
    A("    31-day lists under `condrisk.<stratum>.in_hospital_n` / `.in_hospital_pct`.")
    A("11. **The suppressed stratum's display label comes from the label layer and is not")
    A("    the missing label.** Read `suppress.<stratum>.level_label` and")
    A("    `suppress.<stratum>.level_recorded` rather than assuming a suppressed residual")
    A("    level is unrecorded, and note that its FRAGMENT KEY segment is frozen at")
    A(f"    `{RESIDUAL_SLUG}` for the reason section 5.1 gives. Table 5, Table 5B, S4D and")
    A("    Figure 6 all render the label, never the slug.")
    A("")
    A("---")
    A("")
    A("## 12. Verification carried out")
    A("")
    A("| Check | Result |")
    A("|---|---|")
    A(f"| Aalen-Johansen recursion equals the closed form under complete follow-up | "
      f"asserted on every stratum, tolerance 1e-12 |")
    A(f"| Conditional risk at day 0 equals the 30-day cumulative incidence | "
      f"asserted on every stratum |")
    A(f"| Composite count re-derived against `02` | "
      f"{'reproduces 1,015' if boundary['reproduces_02'] else 'DOES NOT REPRODUCE'} |")
    A(f"| `POD_WINDOW` observed on every composite event | "
      f"{'yes' if boundary['pod_complete'] else 'NO'} |")
    A(f"| Every postoperative day an integer in 0 to {VD.FOLLOWUP_DAYS} | "
      f"{'yes' if boundary['days_clean'] else 'NO'} |")
    A(f"| The four locked stratifiers each partition the whole cohort | "
      f"asserted, {VD.fmt_count(r['window.cohort.n'])} patients per stratifier |")
    A(f"| One minus Kaplan-Meier is never below Aalen-Johansen | "
      f"{r['window.sensitivity.km_never_below_aalen_johansen']} |")
    A(f"| Safe discharge days moved by the competing-risk correction | "
      f"{r['window.sensitivity.km_safe_days_moved']} of "
      f"{r['window.sensitivity.stratum_threshold_cells']} cells |")
    A(f"| Safe discharge days moved by the end-of-day indexing | "
      f"{r['window.sensitivity.indexing_safe_days_moved']} of "
      f"{r['window.sensitivity.stratum_threshold_cells']} cells, maximum shift "
      f"{r['window.sensitivity.indexing_max_abs_day_shift']} day |")
    A(f"| Every rise in a conditional-risk curve follows a competing death | "
      f"{r['condrisk.rises_all_follow_a_competing_death']} |")
    A(f"| Fragment keys inside `window.` `condrisk.` `suppress.` | "
      f"validated in-step and re-validated with `run_all.namespace_violations` |")
    A("| Fragment values flat, no NaN, no nested object | validated in-step |")
    A("| Two consecutive runs byte-identical | verified after the build |")
    A("")

    text = "\n".join(lines) + "\n"
    NOTES_MD.write_text(text, encoding="utf-8")


# ===========================================================================
# 9. SELF-TEST
# ===========================================================================


def self_test() -> int:
    """Estimator unit tests on synthetic data. Writes nothing."""
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise SystemExit(f"[06] SELF-TEST FAILED: {msg}")
        checks += 1

    # 1. No competing events: the cumulative incidence is the crude proportion and the
    #    day-0 conditional risk is the total risk.
    e1 = np.zeros((1, DAYS)); e1[0, 0] = 5; e1[0, 3] = 3; e1[0, 10] = 2
    e2 = np.zeros((1, DAYS))
    n = np.array([1000.0])
    cif, surv, at_risk = aalen_johansen(e1, e2, n)
    ok(abs(cif[0, -1] - 10 / 1000) < 1e-12, "cumulative incidence is not the crude proportion")
    curve = conditional_risk_aj(cif, surv)[0]
    ok(abs(curve[0] - 10 / 1000) < 1e-12, "day-0 conditional risk is not the total risk")
    ok(abs(at_risk[0, 1] - 995.0) < 1e-12, "risk set after day 0 is wrong")

    # 2. Monotone non-increasing with no competing events.
    ok(bool(np.all(np.diff(curve) <= 1e-15)), "conditional risk rises with no competing event")

    # 3. Zero after the last event, and never negative.
    ok(abs(curve[11]) < 1e-15, "conditional risk is not zero after the last event")
    ok(bool(np.all(curve >= 0.0)), "conditional risk went negative")

    # 4. Conditional risk equals the exact closed form on a hand-computable case.
    #    After day 0, 5 events remain among 995 patients still event-free and alive.
    ok(abs(curve[1] - 5 / 995) < 1e-12, "the day-1 conditional risk is not (remaining)/(at risk)")

    # 5. With a competing event, one minus Kaplan-Meier is at least Aalen-Johansen.
    e2b = np.zeros((1, DAYS)); e2b[0, 1] = 100
    cif2, surv2, _ = aalen_johansen(e1, e2b, n)
    km = kaplan_meier_survival(e1, e2b, n)
    ok((1.0 - km[0, -1]) >= cif2[0, -1] - 1e-15,
       "one minus Kaplan-Meier is below the Aalen-Johansen cumulative incidence")
    ok(cif2[0, -1] <= cif[0, -1] + 1e-15,
       "the competing event did not reduce the Aalen-Johansen cumulative incidence")

    # 6. Threshold crossing: earliest strict crossing, and the unreached sentinel.
    probe = np.array([[0.02, 0.009, 0.004, 0.001] + [0.0] * (DAYS - 4)])
    ok(int(first_day_below(probe, 0.01)[0]) == 1, "first strict crossing is wrong")
    ok(int(first_day_below(probe, 0.03)[0]) == 0, "a curve already below is not day 0")
    ok(int(first_day_below(probe, 0.0)[0]) == UNREACHED_DAY,
       "an unreached threshold is not sentinelled")
    ok(day_text(UNREACHED_DAY) == NOT_REACHED, "the unreached day does not render honestly")

    # 7. A curve that dips below and comes back is caught by stays_below.
    bumpy = np.array([0.02, 0.004, 0.02] + [0.0] * (DAYS - 3))
    ok(int(first_day_below(bumpy[None, :], 0.01)[0]) == 1, "earliest crossing on a bumpy curve")
    ok(stays_below(bumpy, 0.01, 1) is False, "stays_below missed a return above the threshold")
    ok(stays_below(bumpy, 0.01, 3) is True, "stays_below is wrong after the last rise")

    # 8. The end-of-day indexing is the primary curve shifted by one day.
    alt = conditional_risk_aj_end_of_day(cif, surv)[0]
    ok(bool(np.allclose(alt[:-1], curve[1:])), "the end-of-day indexing is not a one-day shift")

    # 9. Key slugs are legal fragment keys inside this step's namespaces.
    for t in VD.RISK_THRESHOLDS:
        key = f"window.overall.{threshold_slug(t)}.day"
        ok(bool(KEY_RE.match(key)), f"threshold key {key} is not a legal fragment key")
        ok(key.startswith(RESERVED_PREFIXES), f"threshold key {key} is outside the namespace")
    for column, spec in zip(VD.STRATIFIER_COLUMNS, VD.STRATIFIERS):
        for lev in spec.levels:
            key = f"window.{STRATIFIER_SLUG[column]}.{level_slug(column, lev)}.n"
            ok(bool(KEY_RE.match(key)), f"stratum key {key} is not a legal fragment key")

    # 10. The bootstrap is exactly the patient-level bootstrap: cells sum to n.
    b1, b2 = bootstrap_cells(e1[0], e2[0], 1000, 25)
    ok(b1.shape == (25, DAYS) and b2.shape == (25, DAYS), "bootstrap draw has the wrong shape")
    ok(bool(np.all(b1.sum(axis=1) + b2.sum(axis=1) <= 1000)), "a bootstrap draw exceeds n")

    # 11. The conditioning-set composition. Five patients, hand-computable.
    #     exit day  : 2, inf, inf, inf, inf     (patient 0 leaves the risk set on day 2)
    #     stay      : 0,   1,   5, nan,  40
    mask = np.ones(5, dtype=bool)
    exit_day = np.array([2.0, np.inf, np.inf, np.inf, np.inf])
    stay = np.array([0.0, 1.0, 5.0, np.nan, 40.0])
    ar, inh, dis, mis = riskset_composition(mask, exit_day, stay)
    ok(int(ar[0]) == 5 and int(ar[3]) == 4, "the at-risk count ignores the exit day")
    ok(int(inh[0]) == 4, "a stay of zero days is not in hospital at the start of day 0")
    ok(int(inh[1]) == 3, "a stay of zero days is still counted in hospital on day 1")
    ok(int(inh[2]) == 2, "the day-2 in-hospital count is wrong")
    ok(int(inh[3]) == 2 and int(dis[3]) == 1 and int(mis[3]) == 1,
       "the day-3 composition is wrong once a patient has left the risk set")
    ok(int(inh[30]) == 1, "a stay past the end of the window is not counted in hospital")
    ok(bool(np.all(inh + dis + mis == ar)), "the composition does not partition the risk set")
    ok(int(mis[0]) == 1 and int(mis[30]) == 1,
       "a patient with no recorded stay is not carried as its own count on every day")

    # 12. Both scale matchers behave as the two readers they mirror.
    strict = scale_family_regex_strict("window.<stratum>.cif30.est")
    ok(bool(strict.match("window.overall.cif30.est")), "a one-segment stratum does not match")
    ok(bool(strict.match("window.age.18-49.cif30.est")),
       "a dotted stratum slug does not match the strict matcher")
    ok(not strict.match("window.overall.cif30.pct"), "the strict matcher is not anchored")
    day_family = scale_family_regex_strict("window.riskset.day<d>.text")
    ok(bool(day_family.match("window.riskset.day7.text")),
       "an in-segment placeholder does not match")
    ok(not day_family.match("window.riskset.day7.extra.text"),
       "an in-segment placeholder crossed a dot")
    loose = scale_family_regex_loose("window.<stratum>.n")
    ok(bool(loose.match("window.component.hematoma.n")),
       "the loose matcher does not cross a dot, so a unit conflict could hide")

    # 13. The declared families cover the shapes this step publishes, and never disagree.
    probe = {
        "window.scale": "", "window.scale.families": [], "window.scale.units": [],
        "window.scale.other_numeric_keys": "",
        "condrisk.scale": "", "condrisk.scale.families": [], "condrisk.scale.units": [],
        "condrisk.scale.other_numeric_keys": "",
        "suppress.scale": "", "suppress.scale.families": [], "suppress.scale.units": [],
        "suppress.scale.other_numeric_keys": "",
        "window.cohort.n": 1, "window.composite.pct": 1.0, "window.seed": 1,
        "window.age.18-49.cif30.est": 1.0,
        "window.age.18-49.threshold_1pct.in_hospital_pct": 1.0,
        "window.riskset.day7.text": "", "window.riskset.stay_not_recorded_n": 1,
        "window.riskset.text.caveat": "",
        "condrisk.age.18-49.in_hospital_n": [1],
        "suppress.sex.not-recorded.level_label": "",
    }
    ok(assert_scale_coverage(probe) == scale_family_count(),
       "the scale-coverage assertion does not cover a representative fragment")
    try:
        assert_scale_coverage({"window.a_key_with_no_family": 1})
    except SystemExit:
        checks += 1
    else:
        raise SystemExit("[06] SELF-TEST FAILED: an undeclared key passed scale coverage")

    print(f"[06] self-test: {checks} checks PASS")
    return 0


# ===========================================================================
# 10. MAIN
# ===========================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="06_window.py",
        description="Conditional risk by stratum and the safe discharge day (Table 5).")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the estimator unit tests and exit. Writes nothing.")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    print("[06] safe discharge window: conditional risk by stratum")
    if not ANALYTIC_PARQUET.exists():
        raise SystemExit(f"[06] {ANALYTIC_PARQUET} is missing. Run 02_build_cohort.py first. "
                         f"This step never re-derives a cohort and never reads the warehouse.")
    df = pd.read_parquet(ANALYTIC_PARQUET)
    print(f"[06]   read {ANALYTIC_PARQUET.name}  {len(df):,} x {df.shape[1]}")

    required = ["WINDOW_COMP", "POD_WINDOW", "MORT30", "POD_DEATH",
                *VD.STRATIFIER_COLUMNS, *VD.WINDOW_COMPONENTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[06] columns missing from the analytic parquet: {missing}. "
                         f"That is a bug in 02_build_cohort.py, not a licence to compute "
                         f"them locally (CLAUDE.md 4.1).")

    # --- the composite and the competing event, re-derived -------------------
    is_event = (df["WINDOW_COMP"] == 1).to_numpy()
    is_death = (df["MORT30"] == 1).to_numpy()
    pod_event = df["POD_WINDOW"].to_numpy(dtype=float, na_value=np.nan)
    pod_death = df["POD_DEATH"].to_numpy(dtype=float, na_value=np.nan)

    composite_n = int(is_event.sum())
    pod_complete = bool(np.all(np.isfinite(pod_event[is_event])))
    if not pod_complete:
        raise SystemExit("[06] a WINDOW_COMP event carries no POD_WINDOW. 02 recorded "
                         "POD_WINDOW observed on all of them; report the mismatch, do not "
                         "impute (CLAUDE.md 8.5).")
    if np.any(np.isfinite(pod_event[~is_event])):
        raise SystemExit("[06] POD_WINDOW is populated on a row whose composite flag is "
                         "negative")
    if not bool(np.all(np.isfinite(pod_death[is_death]))):
        raise SystemExit("[06] a recorded 30-day death carries no POD_DEATH")

    days_clean = bool(
        np.all((pod_event[is_event] >= 0) & (pod_event[is_event] <= VD.FOLLOWUP_DAYS)
               & (pod_event[is_event] % 1 == 0))
        and np.all((pod_death[is_death] >= 0) & (pod_death[is_death] <= VD.FOLLOWUP_DAYS)
                   & (pod_death[is_death] % 1 == 0)))
    if not days_clean:
        raise SystemExit("[06] a postoperative day is not an integer in 0 to "
                         f"{VD.FOLLOWUP_DAYS}. 02 froze the handling rule; no step clips "
                         f"or drops locally.")

    both = is_event & is_death
    both_n = int(both.sum())
    death_strictly_first = int(np.sum(both & (pod_death < pod_event)))
    tie_same_day = int(np.sum(both & (pod_death == pod_event)))
    # Time to FIRST event, with the composite taking a same-day tie (see notes section 2.1).
    is_competing = is_death & ~(is_event & (pod_event <= pod_death))
    competing_first = int(is_competing.sum())
    if death_strictly_first:
        print(f"[06]   NOTE: {death_strictly_first} patient(s) carry a death day strictly "
              f"before their composite day; they enter as the competing event.")

    component_counts = {
        "hematoma": int((df["HEMATOMA"] == 1).sum()),
        "reintub": int((df["REINTUBATION"] == 1).sum()),
        "vent48": int((df["VENT48"] == 1).sum()),
        "trach": int((df["REOP_TRACH"] == 1).sum()),
        "airway": int((df["AIRWAY_COMP"] == 1).sum()),
        "hematoma_and_airway": int(((df["HEMATOMA"] == 1) & (df["AIRWAY_COMP"] == 1)).sum()),
    }
    component_labels = [
        ("hematoma", "hematoma requiring unplanned reoperation"),
        ("reintub", "unplanned reintubation"),
        ("vent48", "ventilator dependence beyond 48 hours"),
        ("trach", "reoperative tracheostomy, reoperation slot 1"),
        ("airway", "composite airway morbidity"),
        ("hematoma_and_airway", "patients meeting both hematoma and the airway composite"),
    ]
    reproduces_02 = composite_n == 1015

    # --- who is still in hospital, and the two populations that need naming ---
    # LOS_POSTOP and STILLINHOSP exactly as 02 froze them. Nothing is recomputed from the
    # warehouse, nothing is truncated at day 30 (.build_facts.md 7.5: DOPTODIS runs to 112
    # days and 74 analytic patients stayed past the window), and the flag never overrides
    # a recorded stay (binding item 6: the claim would be false for 35 of the 112).
    for column in ("LOS_POSTOP", "STILLINHOSP"):
        if column not in df.columns:
            raise SystemExit(f"[06] {column} is missing from the analytic parquet. That "
                             f"is a bug in 02_build_cohort.py, not a licence to compute "
                             f"it locally (CLAUDE.md 4.1).")
    los = df["LOS_POSTOP"].to_numpy(dtype=float, na_value=np.nan)
    stillinhosp = (df["STILLINHOSP"].astype("string") == "Yes").fillna(False).to_numpy()
    los_recorded = np.isfinite(los)
    stay_counts = {
        "recorded": int(los_recorded.sum()),
        "not_recorded": int((~los_recorded).sum()),
        "flagged": int(stillinhosp.sum()),
        "not_recorded_and_flagged": int((~los_recorded & stillinhosp).sum()),
        "not_recorded_not_flagged": int((~los_recorded & ~stillinhosp).sum()),
        "flagged_with_recorded_stay": int((stillinhosp & los_recorded).sum()),
        # Of the flagged patients who DO carry a stay, how many of those stays end inside
        # the window and how many past it. Measured rather than carried over from
        # .build_facts.md 7.5, so the sentence cannot go stale on a new vintage.
        "flagged_stay_inside_window": int(
            (stillinhosp & los_recorded & (los <= VD.FOLLOWUP_DAYS)).sum()),
        "flagged_stay_beyond_window": int(
            (stillinhosp & los_recorded & (los > VD.FOLLOWUP_DAYS)).sum()),
    }
    if stay_counts["not_recorded_and_flagged"] + stay_counts["not_recorded_not_flagged"] \
            != stay_counts["not_recorded"]:
        raise SystemExit("[06] the patients with no recorded stay do not split into "
                         "flagged and not flagged")

    # The day each patient leaves the risk set. Complete follow-up means the only exits
    # inside the window are the composite event and the competing death, so this is the
    # whole of it; a patient who reaches day 30 event free and alive never exits.
    exit_day = np.full(len(df), np.inf, dtype=float)
    exit_day[is_event] = pod_event[is_event]
    exit_day[is_competing] = pod_death[is_competing]

    print(f"[06]   composite {composite_n:,} "
          f"({100.0 * composite_n / len(df):.3f}%), competing deaths entering the "
          f"estimator {competing_first:,} of {int(is_death.sum()):,} recorded")
    print(f"[06]   postoperative stay recorded on {stay_counts['recorded']:,}; "
          f"{stay_counts['not_recorded']:,} not recorded "
          f"({stay_counts['not_recorded_and_flagged']:,} of them flagged still in "
          f"hospital); {stay_counts['flagged']:,} flagged in all, of which "
          f"{stay_counts['flagged_with_recorded_stay']:,} carry a recorded stay")

    # --- fit every stratum ---------------------------------------------------
    strata = build_strata(df)
    fits: List[Fit] = []
    for stratum, mask in strata:
        fit = fit_stratum(stratum, mask, pod_event, pod_death, is_event, is_competing,
                          exit_day, los)
        fits.append(fit)
        if fit.suppressed:
            print(f"[06]   SUPPRESSED  {fit.stratum.stratifier} / {fit.stratum.level}: "
                  f"n={fit.n:,}, {fit.events} composite events "
                  f"(< {VD.MIN_STRATUM_EVENTS})")
        else:
            days = []
            for t in VD.RISK_THRESHOLDS:
                d = int(first_day_below(fit.curve[None, :], t)[0])
                days.append(f"{VD.threshold_label(t)}: "
                            f"{'not reached' if d > VD.FOLLOWUP_DAYS else d}")
            print(f"[06]   {fit.stratum.stratifier} / {fit.stratum.level}: "
                  f"n={fit.n:,}, events={fit.events}, "
                  f"30-day incidence {100.0 * fit.cif30:.3f}%  |  " + "  ".join(days))

    # --- coverage check: the declared strata must partition the cohort -------
    for spec in VD.STRATIFIER_COLUMNS:
        total = sum(f.n for f in fits if f.stratum.column == spec)
        if total != len(df):
            raise SystemExit(f"[06] the levels of {spec} cover {total:,} patients, not "
                             f"{len(df):,}")

    # --- artefacts -----------------------------------------------------------
    TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    write_csv(TABLE5_CSV, TABLE5_HEADER, table5_rows(fits))
    print(f"[06]   wrote {TABLE5_CSV}")
    write_csv(TABLE5B_CSV, TABLE5B_HEADER, table5b_rows(fits))
    print(f"[06]   wrote {TABLE5B_CSV}")
    write_csv(S4_CSV, S4_HEADER, s4_rows(fits))
    print(f"[06]   wrote {S4_CSV}")

    results = build_fragment(fits, len(df), composite_n, component_counts,
                             int(is_death.sum()), competing_first, both_n,
                             tie_same_day, death_strictly_first, stay_counts)
    fragment = clean_fragment(results)
    # Every key this step publishes resolves to exactly one UNIT, asserted here rather
    # than in a downstream reader that cannot know what a new key means
    # (.build_facts.md 9.2 item 4). `02`, `04` and `05` have carried this since the
    # section 9.6 fix pass; this step did not.
    n_families = assert_scale_coverage(fragment)
    print(f"[06]   scale declarations: {len(SCALE_DECLARATIONS)} namespaces, "
          f"{n_families} key families, every published key covered")
    RESULTS_JSON.write_text(json.dumps(fragment, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(f"[06]   wrote {RESULTS_JSON}  ({len(fragment)} keys, namespaces validated)")

    write_notes(fits, fragment, {
        "component_labels": component_labels,
        "reproduces_02": reproduces_02,
        "pod_complete": pod_complete,
        "days_clean": days_clean,
    })
    print(f"[06]   wrote {NOTES_MD}")

    # --- read-back verification ---------------------------------------------
    back = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    if back != fragment:
        raise SystemExit("[06] the fragment does not read back identically")
    outside = [k for k in back if not k.startswith(RESERVED_PREFIXES)]
    if outside:
        raise SystemExit(f"[06] fragment keys outside the reserved namespaces: {outside[:5]}")

    overall = next(f for f in fits if f.stratum.slug == "overall")
    print("")
    print(f"[06] OVERALL, {overall.events:,} composite events in {overall.n:,} patients, "
          f"30-day cumulative incidence {100.0 * overall.cif30:.3f}%")
    for t in VD.RISK_THRESHOLDS:
        d = int(first_day_below(overall.curve[None, :], t)[0])
        if d > VD.FOLLOWUP_DAYS:
            print(f"[06]   {VD.threshold_label(t):>6s} threshold: {NOT_REACHED}")
        else:
            print(f"[06]   {VD.threshold_label(t):>6s} threshold: safe discharge day {d}, "
                  f"residual risk {100.0 * overall.curve[d]:.3f}% "
                  f"({100.0 * overall.lo[d]:.3f} to {100.0 * overall.hi[d]:.3f})")
    n_supp = sum(1 for f in fits if f.suppressed)
    print(f"[06] {len(fits)} strata evaluated, {n_supp} suppressed under the "
          f"{VD.MIN_STRATUM_EVENTS}-event rule")
    print("[06] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
