#!/usr/bin/env python3
"""05_inout.py -- postoperative length of stay, the discharge-timing trend, and the
`INOUT` measurement critique.

Owner: T7. Owns `outputs/results_05.json`, `derived/notes_05.md`, and Tables 1, 2, 4, 4b,
4c plus `tables-csv/s4_sensitivity_05.csv`. It writes nothing else.

Inputs (frozen artefacts only; the warehouse is never read and no cohort is re-derived)
--------------------------------------------------------------------------------------
    derived/cohort_analytic.parquet     n = 112,642. THE analysis population.
    derived/cohort_identified.parquet   n = 124,344. RECONCILIATION ONLY (section 6).
    variable_dict.py                    LABELS, SEED, SECTIONS, level maps, formatters
    cpt_acdf.py                         the pinned cohort ladders, imported never retyped

What this step does (PLAN.md 2.1, 2.3, 6.3, 6.4, 6.6, 7.5)
----------------------------------------------------------
    1. Table 1   baseline characteristics by discharge timing (0 / 1 / >= 2 nights).
    2. Table 2   postoperative length of stay and discharge destination by operative year,
                 with the discharge-timing trend tested TWO ways: Cochran-Armitage, and a
                 logistic model carrying operative year as a LINEAR term adjusted for age,
                 sex, body mass index, ASA class and fusion levels.
    3. Table 4   the billing label against true postoperative length of stay: the full 2x2,
                 sensitivity, specificity, predictive values, raw agreement and kappa.
    4. Table 4b  the same, per operative year. THE YEAR-BY-YEAR FORM IS PRIMARY.
    5. Table 4c  the both-ways comparison: one headline complication comparison estimated
                 twice on one cohort with one covariate set, differing only in how the
                 exposure is defined.
    6. Gate 2    every carried-over discharge-timing number re-derived, and the carried-over
                 47,713 reconciled against every candidate denominator.
    7. S4        the trend excluding 2020, and the two length-of-stay fields against each
                 other.

Binding constraints this file is written around
-----------------------------------------------
  * EVERY discharge definition is on the direct days-from-operation-to-discharge field,
    frozen as `LOS_POSTOP`, and NEVER on the billing label. Confusing the two is the
    measurement error this paper documents.
  * `LOS_POSTOP` is NOT window-bounded. It runs to 112 days and 74 analytic patients stayed
    past day 30. The 0-to-30 timing rule applies to days-to-EVENT fields only and must never
    be applied here (.build_facts.md 7.5).
  * `SAMEDAY_DISCH` and `DISCH_BY_POD1` contain patients who did NOT go home: an expired
    destination, a recorded death within 30 days, or a still-in-hospital flag
    (.build_facts.md 7.5 item 8). Both are reported with and without those patients, and the
    exclusion is stated rather than applied silently.
  * Home discharge is defined by the explicit level map in `variable_dict.py`, never by
    substring-matching the word Home, which would fold a facility level into the home group.
  * `NONHOME_DISCH` has an era break at 2021 and carries NO TREND CLAIM (CLAUDE.md 9.6).
    The destination distribution is reported by year because PLAN.md 6.3 requires it; no
    claim that non-home discharge declined is made anywhere.
  * A pooled kappa would hide the two step changes T1 measured in the raw billing column
    (.build_facts.md 7.1, Finding 2), so the year-by-year statistics are primary and the
    pooled statistic is reported only beside them. A change in a billing label is never read
    as a change in practice.
  * THE `INOUT` CRITIQUE IS A REPLICATION, NOT A NEW FINDING (CLAUDE.md 6a.1). It was
    published on ACDF in NSQIP by Bovonratwet et al., Clin Spine Surg 2018, PMID 29351096.
    No key, table cell or note in this step calls it new, novel or first.
  * The both-ways comparison is the ONLY regression whose covariate set is fixed: it is
    `variable_dict.BOTHWAYS_COVARIATES` and nothing else. No discrimination statistic is
    computed anywhere in this step, and no hematoma risk model and no points score is fitted
    (CLAUDE.md 6.6, 7.1).
  * A mismatch against a carried-over number is REPORTED, never overwritten in either
    direction (CLAUDE.md 8.5).

Fragment namespace (CLAUDE.md 4.2): `los.` `trend.` `dest.` `inout.` `bothways.` `baseline.`
Any other prefix is rejected by the merge and stops the pipeline.

Usage
-----
    python3 05_inout.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import cpt_acdf
import variable_dict as VD

# Hard rule 4: seed the global NumPy stream before anything draws on it.
np.random.seed(VD.SEED)

HERE = Path(__file__).resolve().parent
ANALYTIC_PARQUET = HERE / "derived" / "cohort_analytic.parquet"
IDENTIFIED_PARQUET = HERE / "derived" / "cohort_identified.parquet"
TABLES_DIR = HERE / "tables-csv"
OUTPUTS_DIR = HERE / "outputs"
NOTES_PATH = HERE / "derived" / "notes_05.md"
FRAGMENT_PATH = OUTPUTS_DIR / "results_05.json"

RESERVED_PREFIXES: Tuple[str, ...] = (
    "los.", "trend.", "dest.", "inout.", "bothways.", "baseline.",
)

YEARS: Tuple[int, ...] = tuple(range(2012, 2025))
COVID_YEAR = 2020
DEST_ERA_BREAK_YEAR = 2021

# ---------------------------------------------------------------------------
# THE BOTH-WAYS PRE-SPECIFICATION.
#
# Fixed in `derived/notes_05.md` section 0 BEFORE the first fit was run, and reproduced
# here so the constant and the note cannot drift. PLAN.md 6.4 says the size of the shift is
# TO BE MEASURED and forbids asserting a direction in advance; nothing below asserts one.
# ---------------------------------------------------------------------------
BOTHWAYS_OUTCOME = "READMIT_UNPLANNED"
BOTHWAYS_OUTCOME_REASON = (
    "Unplanned readmission within 30 days was fixed before the first fit, for four reasons: "
    "it is the outcome the outpatient spine safety literature leads with, so re-estimating "
    "it under both exposure definitions speaks directly to the published claims the "
    "measurement critique targets; it is the most frequent adverse outcome in this cohort, "
    "so the comparison is powered to show a shift between two definitions rather than "
    "sampling noise; it is ascertained after discharge, so unlike an in-hospital "
    "complication it cannot mechanically lengthen the index stay and thereby help define "
    "the exposure it is regressed on; and it lies entirely outside the companion paper's "
    "reserved results."
)
BOTHWAYS_EXPOSURES: Tuple[Tuple[str, str, str], ...] = (
    ("inout", "INOUT_OUTPT",
     "Outpatient status defined by the billing setting, as the prior literature defines it"),
    ("los", "SAMEDAY_DISCH",
     "Outpatient status defined by a true postoperative length of stay of zero days"),
)

# Continuous variables displayed as median and interquartile range rather than mean and
# standard deviation. Fixed here rather than decided per row: counts and operative time are
# right-skewed and a single stated convention cannot drift between rows.
CONTINUOUS_MEDIAN: frozenset = frozenset({"LEVELS_N", "OPTIME", "LOS_POSTOP"})

# Level display order for the categorical rows of Table 1, taken from variable_dict where
# an ordered list exists. Anything not named here is ordered by its observed levels.
LEVEL_ORDERS: Dict[str, Sequence[str]] = {
    "AGE_CAT": VD.AGE_CAT_ORDER,
    "ASA_CAT": VD.ASA_CAT_ORDER,
    "RACE_CAT": VD.RACE_CAT_ORDER,
    "LEVELS_CAT": VD.LEVELS_CAT_ORDER,
    "DEST_CAT": VD.DEST_CAT_ORDER,
    "SPEC_CAT": VD.SPEC_CAT_ORDER,
    "INOUT": VD.INOUT_ORDER,
    "LOS_CAT": VD.LOS_CAT_ORDER,
}


# ===========================================================================
# 0. Fragment plumbing
# ===========================================================================
class FragmentWriter:
    """Flat, lowercase, dotted keys inside this step's reserved namespace.

    Values are scalars or lists of scalars. NaN is written as null, never as NaN, and a key
    outside the reserved prefixes raises here rather than at the merge.
    """

    def __init__(self, prefixes: Sequence[str]) -> None:
        self._prefixes = tuple(prefixes)
        self._data: Dict[str, Any] = {}

    @staticmethod
    def _clean(value: Any) -> Any:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [FragmentWriter._clean(v) for v in list(value)]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        if isinstance(value, str):
            return value
        raise TypeError(f"fragment value of unsupported type {type(value).__name__}: {value!r}")

    def put(self, key: str, value: Any, decimals: Optional[int] = None) -> None:
        if not any(key.startswith(p) or key == p[:-1] for p in self._prefixes):
            raise ValueError(
                f"fragment key {key!r} is outside this step's reserved namespace "
                f"{list(self._prefixes)} (CLAUDE.md 4.2)"
            )
        if key in self._data:
            raise ValueError(f"fragment key {key!r} written twice inside this step")
        clean = self._clean(value)
        if decimals is not None:
            if isinstance(clean, float):
                clean = round(clean, decimals)
            elif isinstance(clean, list):
                clean = [round(v, decimals) if isinstance(v, float) else v for v in clean]
        self._data[key] = clean

    def __getitem__(self, key: str) -> Any:
        """Read a value back, so the notes RENDER a published number rather than
        recompute one. A number that appears both in the fragment and in prose must be
        the same number, and the only way to guarantee that is to read it back."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> List[str]:
        """Every key published so far, so the coverage assertion can run BEFORE the write."""
        return list(self._data)

    def write(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, sort_keys=True, allow_nan=False)
        path.write_text(payload + "\n", encoding="utf-8")
        return len(self._data)


FRAG = FragmentWriter(RESERVED_PREFIXES)


# ===========================================================================
# 0.1 UNITS AND SCALES (.build_facts.md section 9)
# ===========================================================================
# This step is the worst offender in the merged results file: its `.lo` and `.hi` keys
# are variously PERCENTAGES (agreement, sensitivity, specificity, predictive values),
# ODDS RATIOS (both the trend model and the both-ways comparison) and a KAPPA. Nothing
# in a key name distinguishes them, and `03` publishes the same suffixes as proportions
# while `06` publishes them as percentages. A scale declaration that could be read as
# project-wide is therefore a 100x error waiting to happen, so every declaration below
# names the ONE namespace it governs and says so in its own text.
#
# `<namespace>.scale` is prose a human can read. `<namespace>.scale.families` and
# `<namespace>.scale.units` carry the same content as two parallel lists so
# `verify_numbers.py` can assert a unit mechanically rather than parse English. The
# family entries are KEY PATTERNS addressed to `08`, not rendered prose. No key is
# renamed and none is removed.
#
# CORRECTED 2026-07-31, closing `.build_facts.md` 12.3. The earlier declaration had a hole
# and, worse, two families that OVER-MATCHED a differently-scaled sibling. A wrong
# declaration is more dangerous than a missing one, because a gate trusts it:
#
#   * `inout.by_year.<statistic>` swallowed the year of the smallest and the year of the
#     largest kappa and declared two CALENDAR YEARS as percentages;
#   * `inout.excl.<statistic>` swallowed a Cohen kappa and a count of patients and declared
#     both as percentages, which is the exact 100-fold trap section 9 exists to stop;
#   * `inout.reconcile.<population>.outpatient_night_pct` matched NOTHING at all, because
#     the keys carry a gate segment and a rung segment, so nine reconciliation percentages
#     fell to the catch-all and were described there as counts.
#
# Every prefix-style family that could swallow a differently-scaled sibling has been
# replaced by an enumeration, and `assert_scale_coverage` below now proves mechanically
# that EVERY published key matches exactly one family. That assertion, not the individual
# patches, is what stops this recurring: a key added without a family fails inside this
# step rather than surfacing three steps later.

UNIT_PERCENT = "percentage on the 0 to 100 scale"
UNIT_PP = "percentage points"
UNIT_OR = "odds ratio"
UNIT_KAPPA = "Cohen kappa, minus 1 to 1"
UNIT_DAYS = "days of postoperative stay"
UNIT_COUNT = "count of patients or events"
UNIT_P = "p value"
UNIT_STAT = "test statistic"
UNIT_YEAR = "calendar year"
UNIT_GAPS = "count of year gaps, one fewer than the number of years"
UNIT_LEVELS = "count of distinct discharge destination levels"
UNIT_RUNGS = "count of reconciliation rungs"
UNIT_CANDIDATES = "count of candidate populations tested"
UNIT_ROWS = "count of table rows"
UNIT_PMID = "PubMed identifier, carries no unit"
UNIT_BOOL = "boolean, carries no unit"
UNIT_TEXT = "text, carries no unit"

SCALE_SCOPE_NOTE = (
    "This declaration governs only the namespace it is named for. It is not a "
    "project-wide rule and must never be read as one: the same est, lo and hi "
    "suffixes carry proportions in the timing step and percentages in the window "
    "step, so read the declaration published by the step that owns the key."
)

SCALE_DECLARATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        "inout",
        "Within the billing-label agreement namespace, agreement, disagreement, "
        "sensitivity, specificity and both predictive values are PERCENTAGES on the "
        "0 to 100 scale, and so are their lower and upper limits. Kappa is the one "
        "exception: it and its limits and its standard error are a Cohen kappa on "
        "the minus 1 to 1 scale and are NOT a percentage. Multiplying a kappa by 100 "
        "is the single easiest mistake to make in this namespace, and it is the "
        "reason the year-by-year and went-home families are enumerated key by key "
        "rather than written as a prefix: the smallest and the largest kappa each "
        "carry the YEAR they occurred in, which is a calendar year and not a "
        "percentage, and the went-home sensitivity family holds a kappa and a count "
        "of patients beside its percentages. The reconciliation rungs publish an "
        "outpatient count, a stayed-a-night count and the percentage between them "
        "for every gate and rung, and only the third of those is a percentage.",
        (
            ("inout.agreement", UNIT_PERCENT),
            ("inout.agreement.lo", UNIT_PERCENT),
            ("inout.agreement.hi", UNIT_PERCENT),
            ("inout.disagreement", UNIT_PERCENT),
            ("inout.sensitivity", UNIT_PERCENT),
            ("inout.sensitivity.lo", UNIT_PERCENT),
            ("inout.sensitivity.hi", UNIT_PERCENT),
            ("inout.specificity", UNIT_PERCENT),
            ("inout.specificity.lo", UNIT_PERCENT),
            ("inout.specificity.hi", UNIT_PERCENT),
            ("inout.ppv", UNIT_PERCENT),
            ("inout.ppv.lo", UNIT_PERCENT),
            ("inout.ppv.hi", UNIT_PERCENT),
            ("inout.npv", UNIT_PERCENT),
            ("inout.npv.lo", UNIT_PERCENT),
            ("inout.npv.hi", UNIT_PERCENT),
            ("inout.outpatient_pct", UNIT_PERCENT),
            ("inout.outpatient_night_pct", UNIT_PERCENT),
            ("inout.sameday_pct", UNIT_PERCENT),
            ("inout.kappa", UNIT_KAPPA),
            ("inout.kappa.lo", UNIT_KAPPA),
            ("inout.kappa.hi", UNIT_KAPPA),
            ("inout.kappa.se", UNIT_KAPPA),
            # Year by year. ENUMERATED, never a prefix: this family mixes counts,
            # percentages, a kappa and two calendar years under one stem.
            ("inout.by_year.n", UNIT_COUNT),
            ("inout.by_year.a", UNIT_COUNT),
            ("inout.by_year.b", UNIT_COUNT),
            ("inout.by_year.c", UNIT_COUNT),
            ("inout.by_year.d", UNIT_COUNT),
            ("inout.by_year.outpatient_n", UNIT_COUNT),
            ("inout.by_year.sameday_n", UNIT_COUNT),
            ("inout.by_year.outpatient_night_n", UNIT_COUNT),
            ("inout.by_year.outpatient_pct", UNIT_PERCENT),
            ("inout.by_year.sameday_pct", UNIT_PERCENT),
            ("inout.by_year.outpatient_night_pct", UNIT_PERCENT),
            ("inout.by_year.outpatient_night_pct_min", UNIT_PERCENT),
            ("inout.by_year.outpatient_night_pct_max", UNIT_PERCENT),
            ("inout.by_year.sensitivity", UNIT_PERCENT),
            ("inout.by_year.specificity", UNIT_PERCENT),
            ("inout.by_year.ppv", UNIT_PERCENT),
            ("inout.by_year.npv", UNIT_PERCENT),
            ("inout.by_year.agreement", UNIT_PERCENT),
            ("inout.by_year.disagreement", UNIT_PERCENT),
            ("inout.by_year.disagreement_min", UNIT_PERCENT),
            ("inout.by_year.disagreement_max", UNIT_PERCENT),
            ("inout.by_year.kappa", UNIT_KAPPA),
            ("inout.by_year.kappa.lo", UNIT_KAPPA),
            ("inout.by_year.kappa.hi", UNIT_KAPPA),
            ("inout.by_year.kappa_min", UNIT_KAPPA),
            ("inout.by_year.kappa_max", UNIT_KAPPA),
            ("inout.by_year.kappa_min_year", UNIT_YEAR),
            ("inout.by_year.kappa_max_year", UNIT_YEAR),
            # The same analysis restricted to the patients who went home. Also
            # ENUMERATED: a kappa and a denominator sit beside six percentages.
            ("inout.excl.agreement", UNIT_PERCENT),
            ("inout.excl.sensitivity", UNIT_PERCENT),
            ("inout.excl.specificity", UNIT_PERCENT),
            ("inout.excl.ppv", UNIT_PERCENT),
            ("inout.excl.npv", UNIT_PERCENT),
            ("inout.excl.outpatient_night_pct", UNIT_PERCENT),
            ("inout.excl.kappa", UNIT_KAPPA),
            ("inout.excl.denominator", UNIT_COUNT),
            # The carried-over 47,713 reconciliation, gate by gate and rung by rung.
            ("inout.reconcile.gate_<gate>.<rung>.outpatient_night_pct", UNIT_PERCENT),
            ("inout.reconcile.gate_<gate>.<rung>.outpatient_n", UNIT_COUNT),
            ("inout.reconcile.gate_<gate>.<rung>.outpatient_night_n", UNIT_COUNT),
            ("inout.reconcile.analytic_outpatient_night_pct", UNIT_PERCENT),
            ("inout.reconcile.analytic_outpatient_n", UNIT_COUNT),
            ("inout.reconcile.analytic_outpatient_los_recorded_n", UNIT_COUNT),
            ("inout.reconcile.analytic_outpatient_night_n", UNIT_COUNT),
            ("inout.reconcile.carried_pct", UNIT_PERCENT),
            ("inout.reconcile.carried_numerator", UNIT_COUNT),
            ("inout.reconcile.carried_denominator", UNIT_COUNT),
            ("inout.reconcile.match_pct", UNIT_PERCENT),
            ("inout.reconcile.match_numerator", UNIT_COUNT),
            ("inout.reconcile.match_denominator", UNIT_COUNT),
            ("inout.reconcile.match_numerator_delta", UNIT_COUNT),
            ("inout.reconcile.match_gate", UNIT_TEXT),
            ("inout.reconcile.match_rung", UNIT_TEXT),
            ("inout.reconcile.match_restriction", UNIT_TEXT),
            ("inout.reconcile.candidates_tested", UNIT_CANDIDATES),
            ("inout.reconcile.denominator_matches", UNIT_CANDIDATES),
            ("inout.reconcile.pair_matches", UNIT_CANDIDATES),
            ("inout.reconcile.elective_hypothesis_removes_outpatients", UNIT_COUNT),
            ("inout.reconcile.elective_hypothesis_tested", UNIT_BOOL),
            ("inout.reconcile.elective_hypothesis_holds", UNIT_BOOL),
            ("inout.reconcile.reported_cohort", UNIT_TEXT),
            ("inout.reconcile.note", UNIT_TEXT),
            # The pooled 2x2 and its margins.
            ("inout.denominator", UNIT_COUNT),
            ("inout.excluded_no_los_n", UNIT_COUNT),
            ("inout.cell.outpatient_sameday", UNIT_COUNT),
            ("inout.cell.outpatient_night", UNIT_COUNT),
            ("inout.cell.inpatient_sameday", UNIT_COUNT),
            ("inout.cell.inpatient_night", UNIT_COUNT),
            ("inout.outpatient_n", UNIT_COUNT),
            ("inout.inpatient_n", UNIT_COUNT),
            ("inout.sameday_n", UNIT_COUNT),
            ("inout.night_n", UNIT_COUNT),
            ("inout.outpatient_night_n", UNIT_COUNT),
            ("inout.years", UNIT_YEAR),
            ("inout.step_change_years", UNIT_YEAR),
            ("inout.is_replication", UNIT_BOOL),
            ("inout.pooled_is_secondary", UNIT_BOOL),
            ("inout.prior_art_pmid", UNIT_PMID),
            ("inout.primary_form", UNIT_TEXT),
            ("inout.replication_note", UNIT_TEXT),
            ("inout.step_change_note", UNIT_TEXT),
        ),
    ),
    (
        "bothways",
        "Within the both-ways comparison namespace every estimate and every limit is "
        "an ODDS RATIO on the ratio scale, where 1 is the null. They are NOT "
        "percentages and must never be multiplied by 100. The exposed and unexposed "
        "event rates, the event rate and the discordance are percentages on the 0 to "
        "100 scale. The absolute difference between the two adjusted odds ratios is "
        "on the odds-ratio scale; the relative change beside it is a percentage. "
        "Every plain count in this namespace is a count of PATIENTS except the "
        "event counts, which count outcomes.",
        (
            ("bothways.<definition>.crude.or", UNIT_OR),
            ("bothways.<definition>.crude.lo", UNIT_OR),
            ("bothways.<definition>.crude.hi", UNIT_OR),
            ("bothways.<definition>.adjusted.or", UNIT_OR),
            ("bothways.<definition>.adjusted.lo", UNIT_OR),
            ("bothways.<definition>.adjusted.hi", UNIT_OR),
            ("bothways.shift.or_ratio", UNIT_OR),
            ("bothways.shift.or_difference", UNIT_OR),
            ("bothways.shift.pct_change", UNIT_PERCENT),
            ("bothways.shift.ci_overlap", UNIT_BOOL),
            ("bothways.shift.direction_changed", UNIT_BOOL),
            ("bothways.<definition>.exposed_pct", UNIT_PERCENT),
            ("bothways.<definition>.unexposed_pct", UNIT_PERCENT),
            ("bothways.<definition>.exposed_n", UNIT_COUNT),
            ("bothways.<definition>.unexposed_n", UNIT_COUNT),
            ("bothways.<definition>.exposed_events", UNIT_COUNT),
            ("bothways.<definition>.unexposed_events", UNIT_COUNT),
            ("bothways.<definition>.definition", UNIT_TEXT),
            ("bothways.event_pct", UNIT_PERCENT),
            ("bothways.discordant_pct", UNIT_PERCENT),
            ("bothways.<definition>.<fit>.p", UNIT_P),
            ("bothways.n", UNIT_COUNT),
            ("bothways.events", UNIT_COUNT),
            ("bothways.discordant_n", UNIT_COUNT),
            ("bothways.dropped_incomplete_n", UNIT_COUNT),
            ("bothways.covariates", UNIT_TEXT),
            ("bothways.covariates_identical_between_definitions", UNIT_BOOL),
            ("bothways.no_discrimination_statistic", UNIT_BOOL),
            ("bothways.outcome_label", UNIT_TEXT),
            ("bothways.outcome_prespec_reason", UNIT_TEXT),
            ("bothways.no_difference_test_note", UNIT_TEXT),
        ),
    ),
    (
        "trend",
        "Within the discharge-timing trend namespace the logistic model publishes an "
        "ODDS RATIO per additional operative year, with its two limits on the same "
        "ratio scale; they are not percentages. Everything else is a percentage on "
        "the 0 to 100 scale, except the fitted Cochran-Armitage slope, the fitted "
        "change across the window and the endpoint difference, which are in "
        "PERCENTAGE POINTS. A slope in percentage points per year and an endpoint "
        "difference in percentage points are different quantities and must not be "
        "presented as one another. The first, last and peak year of a series are "
        "CALENDAR YEARS; the percentage that goes with each of them carries the same "
        "stem with a percentage suffix, and the two must never be swapped.",
        (
            ("trend.logit.<definition>.<fit>.or_per_year", UNIT_OR),
            ("trend.logit.<definition>.<fit>.lo", UNIT_OR),
            ("trend.logit.<definition>.<fit>.hi", UNIT_OR),
            ("trend.logit.<definition>.<fit>.p", UNIT_P),
            ("trend.logit.<definition>.<fit>.n", UNIT_COUNT),
            ("trend.logit.<definition>.<fit>.events", UNIT_COUNT),
            ("trend.logit.complete_case_n", UNIT_COUNT),
            ("trend.logit.dropped_incomplete_n", UNIT_COUNT),
            ("trend.logit.covariates", UNIT_TEXT),
            ("trend.logit.year_term", UNIT_TEXT),
            ("trend.logit.no_discrimination_statistic", UNIT_BOOL),
            ("trend.ca.<definition>.<window>.slope_pp_per_year", UNIT_PP),
            ("trend.ca.<definition>.<window>.fitted_change_pp", UNIT_PP),
            ("trend.ca.<definition>.<window>.z", UNIT_STAT),
            ("trend.ca.<definition>.<window>.chi2", UNIT_STAT),
            ("trend.ca.<definition>.<window>.p", UNIT_P),
            ("trend.ca.<definition>.<window>.year_gaps_n", UNIT_GAPS),
            ("trend.<definition>.endpoint_difference_pp", UNIT_PP),
            ("trend.<definition>.pct_by_year", UNIT_PERCENT),
            ("trend.<definition>.first_year_pct", UNIT_PERCENT),
            ("trend.<definition>.last_year_pct", UNIT_PERCENT),
            ("trend.<definition>.peak_pct", UNIT_PERCENT),
            ("trend.<definition>.n_by_year", UNIT_COUNT),
            ("trend.<definition>.first_year", UNIT_YEAR),
            ("trend.<definition>.last_year", UNIT_YEAR),
            ("trend.<definition>.peak_year", UNIT_YEAR),
            ("trend.<definition>.peak_is_an_endpoint", UNIT_BOOL),
            ("trend.<definition>.monotone_increasing", UNIT_BOOL),
            ("trend.<definition>.endpoint_difference_note", UNIT_TEXT),
            ("trend.excl.<definition>.pct_by_year", UNIT_PERCENT),
            ("trend.excl.<definition>.n_by_year", UNIT_COUNT),
            ("trend.excl.denominator_by_year", UNIT_COUNT),
            ("trend.los.<quantile>_by_year", UNIT_DAYS),
            ("trend.patients_by_year", UNIT_COUNT),
            ("trend.denominator_by_year", UNIT_COUNT),
            ("trend.years", UNIT_YEAR),
            ("trend.covid_year", UNIT_YEAR),
            ("trend.covid_shown_not_smoothed", UNIT_BOOL),
            ("trend.covid_note", UNIT_TEXT),
        ),
    ),
    (
        "los",
        "Within the postoperative length-of-stay namespace, a key ending in pct is a "
        "percentage on the 0 to 100 scale. The distribution keys are DAYS of "
        "postoperative stay and are NOT bounded at day 30: the maximum in this cohort "
        "is 112 days and 74 patients stayed past day 30, so the timing rule that "
        "bounds an event day at 30 must never be applied to a length of stay. Two "
        "denominators run in parallel through this namespace: the rate on every "
        "patient whose length of stay is recorded, and the companion rate on the "
        "smaller denominator that excludes the patients who did not go home. Both "
        "are percentages, they are not interchangeable, and each is published with "
        "its own numerator and its own denominator beside it. The reconciliation "
        "rungs carry their own same-day and by-day-1 rates, also percentages, "
        "measured on the pre-exclusion population rather than on the analytic "
        "cohort.",
        (
            ("los.median", UNIT_DAYS),
            ("los.q1", UNIT_DAYS),
            ("los.q3", UNIT_DAYS),
            ("los.min", UNIT_DAYS),
            ("los.max", UNIT_DAYS),
            ("los.analytic_n", UNIT_COUNT),
            ("los.observed_n", UNIT_COUNT),
            ("los.missing_n", UNIT_COUNT),
            ("los.gt30_n", UNIT_COUNT),
            ("los.sameday.pct", UNIT_PERCENT),
            ("los.sameday.excl.pct", UNIT_PERCENT),
            ("los.disch_by_pod1.pct", UNIT_PERCENT),
            ("los.disch_by_pod1.excl.pct", UNIT_PERCENT),
            ("los.sameday.n", UNIT_COUNT),
            ("los.sameday.denominator", UNIT_COUNT),
            ("los.sameday.excl.n", UNIT_COUNT),
            ("los.sameday.excl.denominator", UNIT_COUNT),
            ("los.sameday.notdischargedhome_n", UNIT_COUNT),
            ("los.sameday.expired_dest_n", UNIT_COUNT),
            ("los.sameday.mort30_n", UNIT_COUNT),
            ("los.sameday.stillinhosp_n", UNIT_COUNT),
            ("los.disch_by_pod1.n", UNIT_COUNT),
            ("los.disch_by_pod1.denominator", UNIT_COUNT),
            ("los.disch_by_pod1.excl.n", UNIT_COUNT),
            ("los.disch_by_pod1.excl.denominator", UNIT_COUNT),
            ("los.disch_by_pod1.notdischargedhome_n", UNIT_COUNT),
            ("los.disch_by_pod1.expired_dest_n", UNIT_COUNT),
            ("los.disch_by_pod1.mort30_n", UNIT_COUNT),
            ("los.disch_by_pod1.stillinhosp_n", UNIT_COUNT),
            ("los.notdischargedhome.n", UNIT_COUNT),
            ("los.notdischargedhome.expired_dest_n", UNIT_COUNT),
            ("los.notdischargedhome.mort30_n", UNIT_COUNT),
            ("los.notdischargedhome.stillinhosp_n", UNIT_COUNT),
            ("los.crosscheck.disagree_pct", UNIT_PERCENT),
            ("los.crosscheck.sameday_pct", UNIT_PERCENT),
            ("los.crosscheck.disch_by_pod1_pct", UNIT_PERCENT),
            ("los.crosscheck.delta_min", UNIT_DAYS),
            ("los.crosscheck.delta_max", UNIT_DAYS),
            ("los.crosscheck.both_present_n", UNIT_COUNT),
            ("los.crosscheck.agree_n", UNIT_COUNT),
            ("los.crosscheck.disagree_n", UNIT_COUNT),
            ("los.crosscheck.sameday_n", UNIT_COUNT),
            ("los.crosscheck.disch_by_pod1_n", UNIT_COUNT),
            ("los.crosscheck.parent_both_present_n", UNIT_COUNT),
            ("los.crosscheck.parent_disagree_n", UNIT_COUNT),
            ("los.crosscheck.all_one_day_longer", UNIT_BOOL),
            ("los.crosscheck.reproduces_parent", UNIT_BOOL),
            ("los.gate2.<definition>.<year>.pct", UNIT_PERCENT),
            ("los.gate2.<definition>.<year>.carried_pct", UNIT_PERCENT),
            ("los.gate2.<definition>.<year>.resolved_point_estimate", UNIT_PERCENT),
            ("los.gate2.<definition>.<year>.carried_range_low", UNIT_PERCENT),
            ("los.gate2.<definition>.<year>.carried_range_high", UNIT_PERCENT),
            ("los.gate2.<definition>.<year>.n", UNIT_COUNT),
            ("los.gate2.<definition>.<year>.denominator", UNIT_COUNT),
            ("los.gate2.<definition>.<year>.reproduces", UNIT_BOOL),
            # The median is named for its year with nothing after it, so a trailing
            # placeholder here would swallow the carried value and the verdict flag
            # beside it. Enumerated instead, on the two Gate 2 endpoint years.
            ("los.gate2.median.2012", UNIT_DAYS),
            ("los.gate2.median.2024", UNIT_DAYS),
            ("los.gate2.median.<year>.carried", UNIT_DAYS),
            ("los.gate2.cohort_reported", UNIT_TEXT),
            ("los.gate2.note", UNIT_TEXT),
            ("los.gate2.reconcile.gate_<gate>.<year>.sameday_pct", UNIT_PERCENT),
            ("los.gate2.reconcile.gate_<gate>.<year>.disch_by_pod1_pct", UNIT_PERCENT),
            ("los.gate2.reconcile.gate_<gate>.<year>.median", UNIT_DAYS),
            ("los.gate2.reconcile.gate_<gate>.<year>.denominator", UNIT_COUNT),
            ("los.gate2.reconcile.2024_pod1_closest_pct", UNIT_PERCENT),
            ("los.gate2.reconcile.2024_pod1_closest_delta_pp", UNIT_PP),
            ("los.gate2.reconcile.2024_pod1_closest_rung", UNIT_TEXT),
            ("los.gate2.reconcile.rungs_tested", UNIT_RUNGS),
            ("los.gate2.reconcile.2012_pair_reproduced_n", UNIT_RUNGS),
            ("los.gate2.reconcile.2024_pod1_reproduced_n", UNIT_RUNGS),
            ("los.gate2.reconcile.2012_pair_reproduced_on", UNIT_TEXT),
            ("los.gate2.reconcile.note", UNIT_TEXT),
            ("los.exclusion_rule", UNIT_TEXT),
            ("los.primary_field_note", UNIT_TEXT),
        ),
    ),
    (
        "dest",
        "Within the discharge-destination namespace every key ending in pct is a "
        "percentage on the 0 to 100 scale. The era-break year is a calendar year. "
        "No trend claim may be attached to any of these percentages: the destination "
        "vocabulary changes completely at the era break. The count of distinct "
        "destination levels observed in an era is a count of VOCABULARY LEVELS and "
        "not a count of patients, and it is the measurement that makes the era break "
        "visible.",
        (
            ("dest.<category>.pct_by_year", UNIT_PERCENT),
            ("dest.<category>.n_by_year", UNIT_COUNT),
            ("dest.era.<era>.nonhome_pct", UNIT_PERCENT),
            ("dest.era.<era>.nonhome_n", UNIT_COUNT),
            ("dest.era.<era>.denominator", UNIT_COUNT),
            ("dest.era.<era>.destination_levels", UNIT_LEVELS),
            ("dest.era.<era>.n", UNIT_COUNT),
            ("dest.missing_n_by_year", UNIT_COUNT),
            ("dest.denominator_by_year", UNIT_COUNT),
            ("dest.denominator", UNIT_COUNT),
            ("dest.home_n", UNIT_COUNT),
            ("dest.facility_n", UNIT_COUNT),
            ("dest.other_n", UNIT_COUNT),
            ("dest.missing_n", UNIT_COUNT),
            ("dest.era_break_year", UNIT_YEAR),
            ("dest.years", UNIT_YEAR),
            ("dest.trend_claim_permitted", UNIT_BOOL),
            ("dest.home_definition", UNIT_TEXT),
            ("dest.note", UNIT_TEXT),
        ),
    ),
    (
        "baseline",
        "Within the Table 1 baseline namespace every key ending in pct is a "
        "percentage on the 0 to 100 scale, and it is a share of the analytic cohort, "
        "not of a discharge-timing stratum. The table cells themselves are rendered "
        "strings in the CSV and are authoritative for their own units. The row count "
        "is a count of TABLE ROWS and not of patients.",
        (
            ("baseline.stratum.<level>.pct", UNIT_PERCENT),
            ("baseline.stratum.<level>.n", UNIT_COUNT),
            ("baseline.rows", UNIT_ROWS),
            ("baseline.strata", UNIT_TEXT),
            ("baseline.continuous_median_rule", UNIT_TEXT),
            ("baseline.missing_stratum_handling", UNIT_TEXT),
            ("baseline.test_rule", UNIT_TEXT),
        ),
    ),
)


# The catch-all sentence. Coverage is now total, so nothing reaches it; it stays because
# `07_figures.py` and `verify_numbers.py` read it and because a namespace that publishes a
# family list without a catch-all would look truncated rather than complete.
SCALE_CATCH_ALL = (
    "Every other numeric key in this namespace is a count of patients "
    "or events, a denominator, or a calendar year."
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
        f"[05] SCALE_DECLARATIONS covers {sorted(_DECLARED_NS)} but this step reserves "
        f"{sorted(p.rstrip('.') for p in RESERVED_PREFIXES)}. Every reserved namespace "
        f"declares its own units (.build_facts.md 9.2 item 2)."
    )
if len(set(_DECLARED_NS)) != len(_DECLARED_NS):
    raise SystemExit(f"[05] SCALE_DECLARATIONS declares a namespace twice: {_DECLARED_NS}")


def scale_family_count() -> int:
    """How many key families the six declarations name, in total."""
    return sum(len(families) for _ns, _prose, families in SCALE_DECLARATIONS)


def scale_family_regex(pattern: str) -> "re.Pattern":
    """Turn a family such as `trend.<definition>.pct_by_year` into a matcher.

    A `<...>` placeholder matches WITHIN one dotted segment only, so a wildcard cannot
    silently swallow a deeper key. Identical to the matcher `02_build_cohort.py` and
    `04_postdischarge.py` build, and to the one `07_figures.py` builds from the published
    lists, which is what makes the published families usable by a gate.
    """
    out: List[str] = []
    for part in pattern.split("."):
        seg = ""
        for token in re.split(r"(<[^>]*>)", part):
            seg += r"[^.]*" if token.startswith("<") and token.endswith(">") \
                else re.escape(token)
        out.append(seg)
    return re.compile(r"\.".join(out) + r"$")


def publish_scale_declarations() -> None:
    """Declare the unit of every numeric family this step publishes.

    Binding: `.build_facts.md` section 9. One declaration per namespace, each scoped
    to its own namespace and saying so, because the merged `results.json` resolves
    `.lo` and `.hi` to at least four incompatible scales and three of the four live
    in this step.
    """
    for namespace, prose, families in SCALE_DECLARATIONS:
        FRAG.put(f"{namespace}.scale", f"{prose} {SCALE_SCOPE_NOTE}")
        FRAG.put(f"{namespace}.scale.families", [f for f, _ in families])
        FRAG.put(f"{namespace}.scale.units", [u for _, u in families])
        FRAG.put(f"{namespace}.scale.other_numeric_keys", SCALE_CATCH_ALL)


def assert_scale_coverage(keys: Sequence[str]) -> int:
    """Every published key resolves to exactly one declared family. Raises otherwise.

    This is what makes `.build_facts.md` 9.2 item 4 satisfiable without a special case: a
    gate can ask this fragment for the unit of ANY key it holds and get one answer. A new
    key added without a family fails HERE, inside the step that created it and before the
    fragment is written, rather than in a downstream gate that cannot know what the key
    means. `02_build_cohort.py` and `04_postdischarge.py` have carried this assertion since
    the section 9.6 fix pass; this step did not, which is why `.build_facts.md` 12.3 was
    found by `08` rather than by `05`.

    A key matched by MORE THAN ONE family is a failure too, and deliberately so: that is
    the shape of the defect being closed here, a prefix-style family swallowing a sibling
    on a different scale.
    """
    declared = {ns: [(scale_family_regex(f), f, u) for f, u in fams]
                for ns, _prose, fams in SCALE_DECLARATIONS}
    missing_ns, unmatched, ambiguous = [], [], []
    for key in keys:
        if key.endswith(SCALE_META_SUFFIXES):
            continue
        ns = key.split(".", 1)[0]
        if ns not in declared:
            missing_ns.append(key)
            continue
        hits = [fam for rx, fam, _u in declared[ns] if rx.match(key)]
        if not hits:
            unmatched.append(key)
        elif len(hits) > 1:
            ambiguous.append(f"{key} -> {hits}")
    problems = []
    if missing_ns:
        problems.append(f"in a namespace with no scale declaration: {sorted(missing_ns)[:10]}")
    if unmatched:
        problems.append(f"matched by no declared family: {sorted(unmatched)[:10]}")
    if ambiguous:
        problems.append(f"matched by more than one declared family: {sorted(ambiguous)[:10]}")
    if problems:
        raise SystemExit(
            "[05] scale-declaration coverage gap (.build_facts.md 9.2 item 4):\n       "
            + "\n       ".join(problems)
        )
    return scale_family_count()


def write_table(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write one reviewer-facing CSV, asserting every header and every cell is renderable."""
    for col in columns:
        VD.assert_renderable(col, where=f"{path.name} header")
    if not any("denominator" in c.lower() for c in columns):
        raise ValueError(
            f"{path.name} reports percentages without a denominator column "
            f"(CLAUDE.md hard rule 7)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for row in rows:
            cells = []
            for col in columns:
                cell = row.get(col, "")
                cell = "" if cell is None else str(cell)
                VD.assert_renderable(cell, where=f"{path.name} cell under {col!r}")
                cells.append(cell)
            writer.writerow(cells)


# ===========================================================================
# 1. Statistical helpers
# ===========================================================================
def pct(numerator: float, denominator: float) -> Optional[float]:
    """A percentage on the 0-100 scale, or None when the denominator is zero."""
    if denominator is None or denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion, on the 0-100 scale."""
    if total <= 0:
        return None, None
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def cohen_kappa_2x2(a: int, b: int, c: int, d: int) -> Dict[str, Optional[float]]:
    """Cohen's kappa for a 2x2 agreement table, with the Fleiss large-sample variance.

    Cells are laid out as
        a = outpatient label AND zero postoperative nights
        b = outpatient label AND one or more nights
        c = inpatient label  AND zero postoperative nights
        d = inpatient label  AND one or more nights
    """
    n = a + b + c + d
    if n == 0:
        return {"kappa": None, "se": None, "lo": None, "hi": None,
                "po": None, "pe": None}
    p = np.array([[a, b], [c, d]], dtype=float) / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    po = float(p[0, 0] + p[1, 1])
    pe = float(row[0] * col[0] + row[1] * col[1])
    if abs(1.0 - pe) < 1e-12:
        return {"kappa": None, "se": None, "lo": None, "hi": None, "po": po, "pe": pe}
    kappa = (po - pe) / (1.0 - pe)

    # Fleiss, Cohen and Everitt (1969) large-sample standard error.
    term_a = 0.0
    for i in range(2):
        term_a += p[i, i] * (1.0 - (row[i] + col[i]) * (1.0 - kappa)) ** 2
    term_b = 0.0
    for i in range(2):
        for j in range(2):
            if i == j:
                continue
            term_b += p[i, j] * (col[i] + row[j]) ** 2
    term_b *= (1.0 - kappa) ** 2
    term_c = (kappa - pe * (1.0 - kappa)) ** 2
    var = (term_a + term_b - term_c) / (n * (1.0 - pe) ** 2)
    se = math.sqrt(var) if var > 0 else 0.0
    z = float(stats.norm.ppf(0.975))
    return {"kappa": kappa, "se": se, "lo": kappa - z * se, "hi": kappa + z * se,
            "po": po, "pe": pe}


def cochran_armitage(successes: Sequence[int], totals: Sequence[int],
                     scores: Sequence[float]) -> Dict[str, Optional[float]]:
    """Cochran-Armitage test for trend in a proportion across ordered groups.

    `scores` are the group scores, here the operative year itself, so the test is a test of
    monotone linear trend in the proportion across calendar time.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    t = np.asarray(scores, dtype=float)
    total_n = n.sum()
    total_x = x.sum()
    if total_n == 0 or total_x == 0 or total_x == total_n:
        return {"z": None, "chi2": None, "p": None, "slope_per_year": None}
    p_bar = total_x / total_n
    t_bar = float((n * t).sum() / total_n)
    statistic = float((x * (t - t_bar)).sum())
    variance = float(p_bar * (1.0 - p_bar) * (n * (t - t_bar) ** 2).sum())
    if variance <= 0:
        return {"z": None, "chi2": None, "p": None, "slope_per_year": None}
    z = statistic / math.sqrt(variance)
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    # Weighted least-squares slope of the proportion on the score, reported as the
    # descriptive companion to the test statistic (percentage points per year).
    prop = x / n
    slope = float((n * (t - t_bar) * (prop - p_bar)).sum() / (n * (t - t_bar) ** 2).sum())
    return {"z": z, "chi2": z * z, "p": p_value, "slope_per_year": 100.0 * slope}


def build_design(frame: pd.DataFrame, exposure: str,
                 covariates: Sequence[str]) -> pd.DataFrame:
    """Explicit design matrix. Categorical covariates are dummy coded with a fixed
    reference level so the two both-ways fits cannot differ in their parameterisation."""
    design = pd.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["EXPOSURE"] = frame[exposure].astype(float)
    for name in covariates:
        if name == "LEVELS_CAT":
            for level in list(VD.LEVELS_CAT_ORDER)[1:]:
                design[f"LEVELS.{level}"] = (frame["LEVELS_CAT"] == level).astype(float)
        elif name == "OPERYR":
            for year in YEARS[1:]:
                design[f"YEAR.{year}"] = (frame["OPERYR"].astype("Int64") == year).astype(float)
        else:
            design[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return design


def fit_logit(frame: pd.DataFrame, outcome: str, exposure: str,
              covariates: Sequence[str]) -> Dict[str, Any]:
    """One logistic regression, reporting the EXPOSURE term only.

    No discrimination statistic is computed and no covariate odds ratio is returned
    (CLAUDE.md 6.6): these models exist to adjust an exposure contrast, not to describe risk.
    """
    design = build_design(frame, exposure, covariates)
    y = frame[outcome].astype(float)
    model = sm.Logit(y, design)
    result = model.fit(disp=0, maxiter=200)
    if not result.mle_retvals.get("converged", True):
        raise SystemExit(
            f"logistic model for {outcome} on exposure {exposure} did not converge. "
            f"Stop condition: an unconverged estimate must never be reported."
        )
    beta = float(result.params["EXPOSURE"])
    se = float(result.bse["EXPOSURE"])
    ci = result.conf_int().loc["EXPOSURE"]
    return {
        "n": int(design.shape[0]),
        "events": int(y.sum()),
        "beta": beta,
        "se": se,
        "or": math.exp(beta),
        "lo": math.exp(float(ci[0])),
        "hi": math.exp(float(ci[1])),
        "p": float(result.pvalues["EXPOSURE"]),
        "converged": bool(result.mle_retvals.get("converged", True)),
    }


def fmt_or(fit: Dict[str, Any]) -> str:
    return VD.fmt_or_ci(fit["or"], fit["lo"], fit["hi"])


def fmt_pct_ci(value: Optional[float], lo: Optional[float], hi: Optional[float],
               decimals: int = 1) -> str:
    if value is None:
        return VD.MISSING_LABEL
    if lo is None or hi is None:
        return VD.fmt_pct(value, decimals)
    return f"{VD.fmt_pct(value, decimals)} ({VD.fmt_ci(lo, hi, decimals)})"


# ===========================================================================
# 2. Load the frozen analytic cohort
# ===========================================================================
def load_analytic() -> pd.DataFrame:
    if not ANALYTIC_PARQUET.exists():
        raise SystemExit(
            f"missing {ANALYTIC_PARQUET}. Run 02_build_cohort.py first; this step never "
            f"re-derives a cohort and never reads the warehouse (CLAUDE.md 4.1)."
        )
    frame = pd.read_parquet(ANALYTIC_PARQUET)
    expected = cpt_acdf.COHORT_LADDER_N["analytic"]
    if len(frame) != expected:
        raise SystemExit(
            f"analytic parquet carries {len(frame):,} rows, expected {expected:,} from "
            f"cpt_acdf.COHORT_LADDER_N. Stop condition (CLAUDE.md 8.5)."
        )
    if frame["LOS_POSTOP"].max() <= VD.FOLLOWUP_DAYS:
        raise SystemExit(
            "LOS_POSTOP appears to have been truncated at the follow-up window. It is not "
            "window-bounded and truncating it would corrupt this analysis (.build_facts 7.5)."
        )
    return frame


def not_discharged_home_mask(frame: pd.DataFrame) -> pd.Series:
    """Patients inside `SAMEDAY_DISCH` or `DISCH_BY_POD1` who did not in fact go home.

    Three disjointly-measured conditions, unioned: an expired discharge destination, a
    recorded death within 30 days, or a still-in-hospital flag. `02` publishes the same
    counts under `losfield.*`; they are re-derived here rather than copied
    (.build_facts.md 7.5 binding item 1).
    """
    expired = frame["DISCHDEST"].astype("string").fillna("").eq("Expired")
    died = frame["MORT30"].fillna(0).astype(int).eq(1)
    still_in = frame["STILLINHOSP"].astype("string").fillna("").str.casefold().eq("yes")
    return (expired | died | still_in).astype(bool)


# ===========================================================================
# 3. TABLE 1: baseline characteristics by discharge timing
# ===========================================================================
def _levels_for(frame: pd.DataFrame, column: str) -> List[str]:
    if column in LEVEL_ORDERS:
        ordered = [lv for lv in LEVEL_ORDERS[column]]
    else:
        observed = frame[column].astype("string").fillna("")
        ordered = sorted({v for v in observed.unique() if v != ""})
    return ordered


# Fragment-safe key token for each discharge-timing stratum. A fragment key is a dotted path
# of [a-z0-9_-] segments, so the ">=2" level cannot be written literally; `02` uses the same
# `ge2` spelling under `losfield.los_cat.*`.
STRATUM_KEY_TOKEN: Dict[str, str] = {"0": "0", "1": "1", ">=2": "ge2"}


def _strata(frame: pd.DataFrame) -> List[Tuple[str, str, pd.Series]]:
    """The four Table 1 columns: the three named strata plus the explicit missing stratum.

    `LOS_CAT` is the empty string on 83 analytic rows and `pretty_level` returns the empty
    string rather than a missing label for it (.build_facts.md 7.4 binding item 8), so the
    missing stratum is routed through MISSING_LABEL here rather than rendering blank.
    """
    cat = frame["LOS_CAT"].astype("string").fillna("")
    out: List[Tuple[str, str, pd.Series]] = []
    for level in VD.LOS_CAT_ORDER:
        out.append((STRATUM_KEY_TOKEN[level], VD.pretty_level("LOS_CAT", level),
                    cat.eq(level)))
    out.append(("missing", VD.MISSING_LABEL, cat.eq("")))
    return out


def build_table1(frame: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    strata = _strata(frame)
    stratum_labels = [label for _, label, _ in strata]
    columns = (["Section", "Characteristic", "Level", "Overall"]
               + stratum_labels
               + ["P value", "Denominator overall"]
               + [f"Denominator {label.lower()}" for label in stratum_labels])
    rows: List[Dict[str, Any]] = []

    def emit(section: str, characteristic: str, level: str,
             overall: str, cells: Sequence[str], p_text: str,
             den_overall: int, dens: Sequence[int]) -> None:
        row = {"Section": section, "Characteristic": characteristic, "Level": level,
               "Overall": overall, "P value": p_text,
               "Denominator overall": VD.fmt_count(den_overall)}
        for label, cell in zip(stratum_labels, cells):
            row[label] = cell
        for label, den in zip(stratum_labels, dens):
            row[f"Denominator {label.lower()}"] = VD.fmt_count(den)
        rows.append(row)

    # Header row: the size of each column.
    emit("Cohort", "Number of patients", "",
         VD.fmt_count(len(frame)),
         [VD.fmt_count(int(mask.sum())) for _, _, mask in strata],
         "",
         len(frame),
         [int(mask.sum()) for _, _, mask in strata])

    named = [mask for key, _, mask in strata if key != "missing"]

    for section, names in VD.SECTIONS.items():
        for name in names:
            series = frame[name]
            vtype = VD.var_type(name)
            label = VD.short(name)

            if vtype == "continuous":
                present = series.notna()
                use_median = name in CONTINUOUS_MEDIAN
                stat_label = ("Median (interquartile range)" if use_median
                              else "Mean (standard deviation)")

                def describe(mask: pd.Series) -> str:
                    values = pd.to_numeric(series[mask & present], errors="coerce").dropna()
                    if values.empty:
                        return VD.MISSING_LABEL
                    if use_median:
                        return VD.fmt_iqr(values.median(), values.quantile(0.25),
                                          values.quantile(0.75), decimals=0)
                    return VD.fmt_mean_sd(values.mean(), values.std(ddof=1), decimals=1)

                groups = [pd.to_numeric(series[m & present], errors="coerce").dropna().to_numpy()
                          for m in named]
                groups = [g for g in groups if g.size > 0]
                if len(groups) >= 2:
                    p_value = float(stats.kruskal(*groups).pvalue)
                    p_text = VD.fmt_p(p_value)
                else:
                    p_text = ""
                emit(section, label, stat_label, describe(pd.Series(True, index=frame.index)),
                     [describe(m) for _, _, m in strata], p_text,
                     int(present.sum()), [int((m & present).sum()) for _, _, m in strata])

            elif vtype == "binary":
                present = series.notna()
                positive = series.fillna(0).astype("Int64").eq(1) & present

                def cell(mask: pd.Series) -> str:
                    den = int((mask & present).sum())
                    num = int((mask & positive).sum())
                    return VD.fmt_n_pct(num, den)

                table = np.array([[int((m & positive).sum()),
                                   int((m & present).sum()) - int((m & positive).sum())]
                                  for m in named], dtype=float)
                p_text = _chi2_text(table)
                emit(section, label, VD.pretty_level(name, 1),
                     cell(pd.Series(True, index=frame.index)),
                     [cell(m) for _, _, m in strata], p_text,
                     int(present.sum()), [int((m & present).sum()) for _, _, m in strata])

            else:  # categorical or ordinal
                text = series.astype("string").fillna("")
                present = text.ne("")
                levels = _levels_for(frame, name)
                table = np.array([[int((m & text.eq(lv)).sum()) for lv in levels]
                                  for m in named], dtype=float)
                p_text = _chi2_text(table)
                for index, level in enumerate(levels):
                    hit = text.eq(level)

                    def cell(mask: pd.Series, hit=hit) -> str:
                        den = int((mask & present).sum())
                        return VD.fmt_n_pct(int((mask & hit).sum()), den)

                    emit(section, label if index == 0 else "",
                         VD.pretty_level(name, level),
                         cell(pd.Series(True, index=frame.index)),
                         [cell(m) for _, _, m in strata],
                         p_text if index == 0 else "",
                         int(present.sum()),
                         [int((m & present).sum()) for _, _, m in strata])
                missing_n = int((~present).sum())
                if missing_n:
                    def cell_missing(mask: pd.Series) -> str:
                        return VD.fmt_count(int((mask & ~present).sum()))

                    emit(section, "", VD.MISSING_LABEL,
                         VD.fmt_count(missing_n),
                         [cell_missing(m) for _, _, m in strata], "",
                         len(frame), [int(m.sum()) for _, _, m in strata])
    return columns, rows


def _chi2_text(table: np.ndarray) -> str:
    """Chi-square test of independence across the three named discharge-timing strata."""
    keep = table[:, table.sum(axis=0) > 0]
    keep = keep[keep.sum(axis=1) > 0, :]
    if keep.shape[0] < 2 or keep.shape[1] < 2:
        return ""
    try:
        p_value = float(stats.chi2_contingency(keep).pvalue)
    except ValueError:
        return ""
    return VD.fmt_p(p_value)


# ===========================================================================
# 4. TABLE 2: length of stay and destination by year, and the trend
# ===========================================================================
def build_table2(frame: pd.DataFrame, excluded: pd.Series) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    los = frame["LOS_POSTOP"]
    observed = los.notna()
    keep = observed & ~excluded
    dest = frame["DEST_CAT"].astype("string").fillna("")
    dest_present = dest.ne("")

    columns = [
        "Operative year", "Patients",
        "Same day discharge, n (%)",
        "Discharge by postoperative day 1, n (%)",
        "Postoperative stay, median (interquartile range), days",
        "Denominator, length of stay recorded",
        "Same day discharge excluding patients not discharged home, n (%)",
        "Discharge by postoperative day 1 excluding patients not discharged home, n (%)",
        "Denominator, length of stay recorded and discharged home or alive",
        "Home, n (%)", "Facility, n (%)", "Other destination, n (%)",
        "Destination not recorded, n",
        "Denominator, destination recorded",
    ]
    rows: List[Dict[str, Any]] = []
    series: Dict[str, List[Any]] = {k: [] for k in (
        "years", "n", "los_den", "sameday_n", "sameday_pct", "pod1_n", "pod1_pct",
        "los_median", "los_q1", "los_q3",
        "clean_den", "sameday_clean_n", "sameday_clean_pct", "pod1_clean_n", "pod1_clean_pct",
        "dest_den", "home_n", "home_pct", "facility_n", "facility_pct",
        "other_n", "other_pct", "dest_missing_n",
    )}

    def emit(label: str, mask: pd.Series, record: bool) -> None:
        den = int((mask & observed).sum())
        den_clean = int((mask & keep).sum())
        den_dest = int((mask & dest_present).sum())
        sameday = int((mask & observed & los.eq(0)).sum())
        pod1 = int((mask & observed & los.le(1)).sum())
        sameday_clean = int((mask & keep & los.eq(0)).sum())
        pod1_clean = int((mask & keep & los.le(1)).sum())
        values = pd.to_numeric(los[mask & observed], errors="coerce").dropna()
        home = int((mask & dest.eq("Home")).sum())
        facility = int((mask & dest.eq("Facility")).sum())
        other = int((mask & dest.eq("Other")).sum())
        dest_missing = int((mask & ~dest_present).sum())
        rows.append({
            "Operative year": label,
            "Patients": VD.fmt_count(int(mask.sum())),
            "Same day discharge, n (%)": VD.fmt_n_pct(sameday, den),
            "Discharge by postoperative day 1, n (%)": VD.fmt_n_pct(pod1, den),
            "Postoperative stay, median (interquartile range), days":
                VD.fmt_iqr(values.median(), values.quantile(0.25), values.quantile(0.75), 0),
            "Denominator, length of stay recorded": VD.fmt_count(den),
            "Same day discharge excluding patients not discharged home, n (%)":
                VD.fmt_n_pct(sameday_clean, den_clean),
            "Discharge by postoperative day 1 excluding patients not discharged home, n (%)":
                VD.fmt_n_pct(pod1_clean, den_clean),
            "Denominator, length of stay recorded and discharged home or alive":
                VD.fmt_count(den_clean),
            "Home, n (%)": VD.fmt_n_pct(home, den_dest),
            "Facility, n (%)": VD.fmt_n_pct(facility, den_dest),
            "Other destination, n (%)": VD.fmt_n_pct(other, den_dest),
            "Destination not recorded, n": VD.fmt_count(dest_missing),
            "Denominator, destination recorded": VD.fmt_count(den_dest),
        })
        if record:
            series["years"].append(int(label))
            series["n"].append(int(mask.sum()))
            series["los_den"].append(den)
            series["sameday_n"].append(sameday)
            series["sameday_pct"].append(pct(sameday, den))
            series["pod1_n"].append(pod1)
            series["pod1_pct"].append(pct(pod1, den))
            series["los_median"].append(float(values.median()))
            series["los_q1"].append(float(values.quantile(0.25)))
            series["los_q3"].append(float(values.quantile(0.75)))
            series["clean_den"].append(den_clean)
            series["sameday_clean_n"].append(sameday_clean)
            series["sameday_clean_pct"].append(pct(sameday_clean, den_clean))
            series["pod1_clean_n"].append(pod1_clean)
            series["pod1_clean_pct"].append(pct(pod1_clean, den_clean))
            series["dest_den"].append(den_dest)
            series["home_n"].append(home)
            series["home_pct"].append(pct(home, den_dest))
            series["facility_n"].append(facility)
            series["facility_pct"].append(pct(facility, den_dest))
            series["other_n"].append(other)
            series["other_pct"].append(pct(other, den_dest))
            series["dest_missing_n"].append(dest_missing)

    year = frame["OPERYR"].astype("Int64")
    for value in YEARS:
        emit(str(value), year.eq(value), record=True)
    emit("All years", pd.Series(True, index=frame.index), record=False)
    rows[-1]["Operative year"] = "All years"
    return columns, rows, series


# ===========================================================================
# 5. TABLE 4 and 4b: the billing label against true postoperative length of stay
# ===========================================================================
def agreement_stats(outpatient: pd.Series, sameday: pd.Series) -> Dict[str, Any]:
    """The 2x2 and every statistic recomputable from its four cells."""
    a = int((outpatient & sameday).sum())          # label outpatient, 0 nights
    b = int((outpatient & ~sameday).sum())         # label outpatient, 1 or more nights
    c = int((~outpatient & sameday).sum())         # label inpatient, 0 nights
    d = int((~outpatient & ~sameday).sum())        # label inpatient, 1 or more nights
    n = a + b + c + d
    kappa = cohen_kappa_2x2(a, b, c, d)
    sens = pct(a, a + c)
    spec = pct(d, b + d)
    ppv = pct(a, a + b)
    npv = pct(d, c + d)
    agree = pct(a + d, n)
    return {
        "a": a, "b": b, "c": c, "d": d, "n": n,
        "outpatient_n": a + b, "inpatient_n": c + d,
        "sameday_n": a + c, "night_n": b + d,
        "sensitivity": sens, "sensitivity_ci": wilson_ci(a, a + c),
        "specificity": spec, "specificity_ci": wilson_ci(d, b + d),
        "ppv": ppv, "ppv_ci": wilson_ci(a, a + b),
        "npv": npv, "npv_ci": wilson_ci(d, c + d),
        "agreement": agree, "agreement_ci": wilson_ci(a + d, n),
        "disagreement": None if agree is None else 100.0 - agree,
        "kappa": kappa["kappa"], "kappa_se": kappa["se"],
        "kappa_lo": kappa["lo"], "kappa_hi": kappa["hi"],
        "outpatient_night_pct": pct(b, a + b),
        "outpatient_pct": pct(a + b, n),
        "sameday_pct": pct(a + c, n),
    }


def build_table4(stats_all: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    columns = ["Measure", "Definition", "n", "Denominator", "Estimate",
               "95% confidence interval"]
    rows: List[Dict[str, Any]] = []

    def cell(measure: str, definition: str, n: Any, den: Any,
             estimate: str, ci: str = "") -> None:
        rows.append({"Measure": measure, "Definition": definition,
                     "n": "" if n is None else VD.fmt_count(n),
                     "Denominator": "" if den is None else VD.fmt_count(den),
                     "Estimate": estimate, "95% confidence interval": ci})

    n = stats_all["n"]
    cell("Cross-tabulation", "Outpatient billing setting and zero postoperative nights",
         stats_all["a"], n, VD.fmt_pct(pct(stats_all["a"], n)))
    cell("Cross-tabulation", "Outpatient billing setting and one or more nights",
         stats_all["b"], n, VD.fmt_pct(pct(stats_all["b"], n)))
    cell("Cross-tabulation", "Inpatient billing setting and zero postoperative nights",
         stats_all["c"], n, VD.fmt_pct(pct(stats_all["c"], n)))
    cell("Cross-tabulation", "Inpatient billing setting and one or more nights",
         stats_all["d"], n, VD.fmt_pct(pct(stats_all["d"], n)))
    cell("Margin", "Outpatient billing setting, all", stats_all["outpatient_n"], n,
         VD.fmt_pct(pct(stats_all["outpatient_n"], n)))
    cell("Margin", "Inpatient billing setting, all", stats_all["inpatient_n"], n,
         VD.fmt_pct(pct(stats_all["inpatient_n"], n)))
    cell("Margin", "Zero postoperative nights, all", stats_all["sameday_n"], n,
         VD.fmt_pct(pct(stats_all["sameday_n"], n)))
    cell("Margin", "One or more postoperative nights, all", stats_all["night_n"], n,
         VD.fmt_pct(pct(stats_all["night_n"], n)))
    cell("Headline disagreement",
         "Cases labelled outpatient that stayed at least one night",
         stats_all["b"], stats_all["outpatient_n"],
         VD.fmt_pct(stats_all["outpatient_night_pct"]))
    cell("Sensitivity",
         "Outpatient label among patients truly discharged the same day",
         stats_all["a"], stats_all["sameday_n"], VD.fmt_pct(stats_all["sensitivity"]),
         VD.fmt_ci(*stats_all["sensitivity_ci"], decimals=1))
    cell("Specificity",
         "Inpatient label among patients who stayed at least one night",
         stats_all["d"], stats_all["night_n"], VD.fmt_pct(stats_all["specificity"]),
         VD.fmt_ci(*stats_all["specificity_ci"], decimals=1))
    cell("Positive predictive value",
         "Truly same day among patients labelled outpatient",
         stats_all["a"], stats_all["outpatient_n"], VD.fmt_pct(stats_all["ppv"]),
         VD.fmt_ci(*stats_all["ppv_ci"], decimals=1))
    cell("Negative predictive value",
         "Stayed at least one night among patients labelled inpatient",
         stats_all["d"], stats_all["inpatient_n"], VD.fmt_pct(stats_all["npv"]),
         VD.fmt_ci(*stats_all["npv_ci"], decimals=1))
    cell("Raw agreement", "Label and length of stay agree",
         stats_all["a"] + stats_all["d"], n, VD.fmt_pct(stats_all["agreement"]),
         VD.fmt_ci(*stats_all["agreement_ci"], decimals=1))
    cell("Disagreement", "Label and length of stay disagree",
         stats_all["b"] + stats_all["c"], n, VD.fmt_pct(stats_all["disagreement"]))
    cell("Cohen kappa", "Chance-corrected agreement", None, n,
         f"{stats_all['kappa']:.3f}",
         VD.fmt_ci(stats_all["kappa_lo"], stats_all["kappa_hi"], decimals=3))
    return columns, rows


def build_table4b(frame: pd.DataFrame, outpatient: pd.Series, sameday: pd.Series,
                  usable: pd.Series, stats_all: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[Any]]]:
    columns = [
        "Operative year", "Outpatient billing setting, n (%)",
        "Zero postoperative nights, n (%)",
        "Outpatient and zero nights, n", "Outpatient and one or more nights, n",
        "Inpatient and zero nights, n", "Inpatient and one or more nights, n",
        "Cases labelled outpatient that stayed at least one night, n (%)",
        "Sensitivity, %", "Specificity, %",
        "Positive predictive value, %", "Negative predictive value, %",
        "Raw agreement, %", "Disagreement, %", "Cohen kappa",
        "Denominator, both fields recorded",
    ]
    rows: List[Dict[str, Any]] = []
    per_year: Dict[str, List[Any]] = {k: [] for k in (
        "years", "n", "a", "b", "c", "d", "outpatient_n", "outpatient_pct",
        "sameday_n", "sameday_pct", "outpatient_night_n", "outpatient_night_pct",
        "sensitivity", "specificity", "ppv", "npv", "agreement", "disagreement",
        "kappa", "kappa_lo", "kappa_hi",
    )}
    year = frame["OPERYR"].astype("Int64")

    def emit(label: str, stat: Dict[str, Any]) -> None:
        rows.append({
            "Operative year": label,
            "Outpatient billing setting, n (%)":
                VD.fmt_n_pct(stat["outpatient_n"], stat["n"]),
            "Zero postoperative nights, n (%)": VD.fmt_n_pct(stat["sameday_n"], stat["n"]),
            "Outpatient and zero nights, n": VD.fmt_count(stat["a"]),
            "Outpatient and one or more nights, n": VD.fmt_count(stat["b"]),
            "Inpatient and zero nights, n": VD.fmt_count(stat["c"]),
            "Inpatient and one or more nights, n": VD.fmt_count(stat["d"]),
            "Cases labelled outpatient that stayed at least one night, n (%)":
                VD.fmt_n_pct(stat["b"], stat["outpatient_n"]),
            "Sensitivity, %": VD.fmt_pct(stat["sensitivity"]),
            "Specificity, %": VD.fmt_pct(stat["specificity"]),
            "Positive predictive value, %": VD.fmt_pct(stat["ppv"]),
            "Negative predictive value, %": VD.fmt_pct(stat["npv"]),
            "Raw agreement, %": VD.fmt_pct(stat["agreement"]),
            "Disagreement, %": VD.fmt_pct(stat["disagreement"]),
            "Cohen kappa": f"{stat['kappa']:.3f}",
            "Denominator, both fields recorded": VD.fmt_count(stat["n"]),
        })

    for value in YEARS:
        mask = usable & year.eq(value)
        stat = agreement_stats(outpatient[mask], sameday[mask])
        emit(str(value), stat)
        per_year["years"].append(int(value))
        for key in ("n", "a", "b", "c", "d", "outpatient_n", "sameday_n"):
            per_year[key].append(stat[key])
        per_year["outpatient_night_n"].append(stat["b"])
        for key in ("outpatient_pct", "sameday_pct", "outpatient_night_pct",
                    "sensitivity", "specificity", "ppv", "npv", "agreement",
                    "disagreement", "kappa"):
            per_year[key].append(stat[key])
        per_year["kappa_lo"].append(stat["kappa_lo"])
        per_year["kappa_hi"].append(stat["kappa_hi"])
    emit("All years, pooled", stats_all)
    return columns, rows, per_year


# ===========================================================================
# 6. TABLE 4c: the both-ways comparison
# ===========================================================================
def build_bothways(frame: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    covariates = list(VD.BOTHWAYS_COVARIATES)
    needed = [BOTHWAYS_OUTCOME, "INOUT_OUTPT", "SAMEDAY_DISCH", "OPERYR", "LEVELS_CAT"]
    numeric_cov = [c for c in covariates if c not in ("LEVELS_CAT", "OPERYR")]
    complete = pd.Series(True, index=frame.index)
    for column in needed + numeric_cov:
        if frame[column].dtype == object or str(frame[column].dtype) == "string":
            complete &= frame[column].astype("string").fillna("").ne("")
        else:
            complete &= frame[column].notna()
    analysis = frame[complete].copy()
    dropped = int(len(frame) - len(analysis))

    outcome = analysis[BOTHWAYS_OUTCOME].astype(int)
    results: Dict[str, Dict[str, Any]] = {}
    for key, exposure, _ in BOTHWAYS_EXPOSURES:
        exposed = analysis[exposure].astype(int).eq(1)
        crude = fit_logit(analysis.assign(**{exposure: analysis[exposure].astype(float)}),
                          BOTHWAYS_OUTCOME, exposure, [])
        adjusted = fit_logit(analysis, BOTHWAYS_OUTCOME, exposure, covariates)
        results[key] = {
            "exposure": exposure,
            "exposed_n": int(exposed.sum()),
            "exposed_events": int(outcome[exposed].sum()),
            "unexposed_n": int((~exposed).sum()),
            "unexposed_events": int(outcome[~exposed].sum()),
            "crude": crude,
            "adjusted": adjusted,
        }

    inout_exposed = analysis["INOUT_OUTPT"].astype(int).eq(1)
    los_exposed = analysis["SAMEDAY_DISCH"].astype(int).eq(1)
    discordant = int((inout_exposed ^ los_exposed).sum())

    or_inout = results["inout"]["adjusted"]["or"]
    or_los = results["los"]["adjusted"]["or"]
    shift = {
        "or_ratio": or_los / or_inout,
        "or_difference": or_los - or_inout,
        "pct_change": 100.0 * (or_los - or_inout) / or_inout,
        "ci_overlap": bool(results["inout"]["adjusted"]["lo"] <= results["los"]["adjusted"]["hi"]
                           and results["los"]["adjusted"]["lo"] <= results["inout"]["adjusted"]["hi"]),
        "direction_changed": bool((or_inout - 1.0) * (or_los - 1.0) < 0),
        "discordant_n": discordant,
        "discordant_pct": pct(discordant, len(analysis)),
        "n": int(len(analysis)),
        "dropped": dropped,
        "events": int(outcome.sum()),
    }

    columns = ["Exposure definition", "Model", "Exposed, n", "Exposed with the outcome, n",
               "Exposed rate, %", "Unexposed, n", "Unexposed with the outcome, n",
               "Unexposed rate, %", "Odds ratio (95% confidence interval)", "P value",
               "Denominator"]
    rows: List[Dict[str, Any]] = []
    for key, exposure, definition in BOTHWAYS_EXPOSURES:
        record = results[key]
        for model_label, fit in (("Unadjusted", record["crude"]),
                                 ("Adjusted", record["adjusted"])):
            rows.append({
                "Exposure definition": definition,
                "Model": model_label,
                "Exposed, n": VD.fmt_count(record["exposed_n"]),
                "Exposed with the outcome, n": VD.fmt_count(record["exposed_events"]),
                "Exposed rate, %": VD.fmt_pct(pct(record["exposed_events"], record["exposed_n"]), 2),
                "Unexposed, n": VD.fmt_count(record["unexposed_n"]),
                "Unexposed with the outcome, n": VD.fmt_count(record["unexposed_events"]),
                "Unexposed rate, %": VD.fmt_pct(pct(record["unexposed_events"],
                                                    record["unexposed_n"]), 2),
                "Odds ratio (95% confidence interval)": fmt_or(fit),
                "P value": VD.fmt_p(fit["p"]),
                "Denominator": VD.fmt_count(shift["n"]),
            })
    rows.append({
        "Exposure definition": "Shift attributable to the definition change",
        "Model": "Adjusted length of stay estimate divided by adjusted billing estimate",
        "Exposed, n": "", "Exposed with the outcome, n": "", "Exposed rate, %": "",
        "Unexposed, n": "", "Unexposed with the outcome, n": "", "Unexposed rate, %": "",
        "Odds ratio (95% confidence interval)": f"{shift['or_ratio']:.2f}",
        "P value": "", "Denominator": VD.fmt_count(shift["n"]),
    })
    rows.append({
        "Exposure definition": "Patients classified differently by the two definitions",
        "Model": "Discordant exposure classification",
        "Exposed, n": VD.fmt_count(discordant),
        "Exposed with the outcome, n": "",
        "Exposed rate, %": VD.fmt_pct(shift["discordant_pct"]),
        "Unexposed, n": "", "Unexposed with the outcome, n": "", "Unexposed rate, %": "",
        "Odds ratio (95% confidence interval)": "",
        "P value": "", "Denominator": VD.fmt_count(shift["n"]),
    })
    return columns, rows, {"results": results, "shift": shift}


# ===========================================================================
# 7. Gate 2 and the 47,713 reconciliation
# ===========================================================================
def load_identified_for_reconciliation() -> Tuple[pd.DataFrame, pd.Series]:
    """The pre-exclusion frame, read ONCE and used ONLY for reconciliation.

    RECONCILIATION ONLY. This is the one place in this step that reads the pre-exclusion
    parquet, and no figure it produces is a denominator for a reported percentage:
    CLAUDE.md 4.4 rules that the manuscript reports the billing-label analysis on the
    ANALYTIC cohort. Nothing here re-derives a cohort or reads the warehouse; the
    pre-exclusion frame and the ladder masks were both frozen by 02, and the ladder filter
    is asserted against the pinned constant rather than trusted.
    """
    frame = pd.read_parquet(IDENTIFIED_PARQUET)
    expected = cpt_acdf.COHORT_LADDER_N["identified"]
    if len(frame) != expected:
        raise SystemExit(
            f"pre-exclusion parquet carries {len(frame):,} rows, expected {expected:,}."
        )
    nonagenarian = frame["AGE"].astype("Int64").eq(90)
    delta = cpt_acdf.COHORT_LADDER_NONAGENARIAN_DELTA["identified"]
    if int(nonagenarian.sum()) != delta:
        raise SystemExit(
            f"the reconciliation ladder filter selects {int(nonagenarian.sum())} top-coded "
            f"age rows, expected {delta} from cpt_acdf. Stop condition."
        )
    return frame, nonagenarian


def gate2_on_identified(frame: pd.DataFrame, nonagenarian: pd.Series) -> List[Dict[str, Any]]:
    """The carried-over discharge-timing figures, measured on the pre-exclusion frame.

    RECONCILIATION ONLY. `CLAUDE.md` 4.4 rules that the manuscript reports the analytic
    cohort; this exists so a mismatch can be explained rather than merely announced.
    """
    los = frame["LOS_POSTOP"]
    observed = los.notna()
    year = frame["OPERYR"].astype("Int64")
    out: List[Dict[str, Any]] = []
    for gate_key, gate_mask in (("b", pd.Series(True, index=frame.index)),
                                ("a", ~nonagenarian)):
        for rung_key, rung_mask in (
                ("identified", pd.Series(True, index=frame.index)),
                ("elective", frame["STEP_ELECTIVE"].fillna(0).astype(int).eq(1)),
                ("exclusions", frame["STEP_EXCLUSIONS"].fillna(0).astype(int).eq(1))):
            for year_value in (2012, 2024):
                mask = gate_mask & rung_mask & observed & year.eq(year_value)
                den = int(mask.sum())
                sd = int((mask & los.eq(0)).sum())
                p1 = int((mask & los.le(1)).sum())
                values = pd.to_numeric(los[mask], errors="coerce").dropna()
                out.append({
                    "gate": gate_key, "rung": rung_key, "year": year_value,
                    "denominator": den, "sameday_n": sd, "sameday_pct": pct(sd, den),
                    "pod1_n": p1, "pod1_pct": pct(p1, den),
                    "median": float(values.median()) if den else None,
                })
    return out


def reconcile_carryover_outpatient_count(frame: pd.DataFrame,
                                         nonagenarian: pd.Series) -> Dict[str, Any]:
    """Resolve the denominator behind the carried-over 34,879 of 47,713."""
    outpatient = frame["INOUT"].astype("string").fillna("").eq(VD.INOUT_OUTPATIENT_LEVEL)
    los = frame["LOS_POSTOP"]
    check_present = frame["LOS_POSTOP_CHECK"].notna()

    rungs = {
        "identified": pd.Series(True, index=frame.index),
        "elective": frame["STEP_ELECTIVE"].fillna(0).astype(int).eq(1),
        "exclusions": frame["STEP_EXCLUSIONS"].fillna(0).astype(int).eq(1),
        "analytic": frame["STEP_ANALYTIC"].fillna(0).astype(int).eq(1),
    }
    gates = {
        "b": pd.Series(True, index=frame.index),
        "a": ~nonagenarian,
    }
    # Keys are fragment-safe tokens; the label is the renderable description a table or the
    # manuscript would print, so no machine token can leak into rendered text.
    restrictions = {
        "all": (pd.Series(True, index=frame.index), "no restriction"),
        "losrecorded": (los.notna(), "postoperative length of stay recorded"),
        "crosscheck": (check_present, "cross-check length of stay computable"),
    }

    candidates: List[Dict[str, Any]] = []
    for gate_key, gate_mask in gates.items():
        for rung_key, rung_mask in rungs.items():
            for restriction_key, (restriction_mask, restriction_label) in restrictions.items():
                mask = gate_mask & rung_mask & outpatient & restriction_mask
                denominator = int(mask.sum())
                numerator = int((mask & los.gt(0)).sum())
                candidates.append({
                    "gate": gate_key, "rung": rung_key, "restriction": restriction_key,
                    "restriction_label": restriction_label,
                    "denominator": denominator, "numerator": numerator,
                    "pct": pct(numerator, denominator),
                    "denominator_match": denominator == 47713,
                    "pair_match": denominator == 47713 and numerator == 34879,
                })
    return {"frame_n": int(len(frame)), "candidates": candidates}


# ===========================================================================
# 8. main
# ===========================================================================
def main() -> int:
    print("05_inout.py -- length of stay, the discharge-timing trend, and the billing-label")
    print("               measurement critique (a REPLICATION of PMID 29351096, never new)")
    print()

    publish_scale_declarations()
    print(f"scale declarations {len(SCALE_DECLARATIONS)} namespaces, "
          f"{scale_family_count()} families, each scoped to itself "
          f"(.build_facts.md 9)")

    frame = load_analytic()
    analytic_n = len(frame)
    print(f"analytic cohort {analytic_n:,} rows x {frame.shape[1]} columns")

    los = frame["LOS_POSTOP"]
    los_observed = los.notna()
    excluded = not_discharged_home_mask(frame)
    clean = los_observed & ~excluded

    # -------------------------------------------------------------------
    # Pooled length-of-stay description and the contamination counts.
    # -------------------------------------------------------------------
    values = pd.to_numeric(los[los_observed], errors="coerce").dropna()
    FRAG.put("los.analytic_n", analytic_n)
    FRAG.put("los.observed_n", int(los_observed.sum()))
    FRAG.put("los.missing_n", int((~los_observed).sum()))
    FRAG.put("los.median", float(values.median()))
    FRAG.put("los.q1", float(values.quantile(0.25)))
    FRAG.put("los.q3", float(values.quantile(0.75)))
    FRAG.put("los.min", float(values.min()))
    FRAG.put("los.max", float(values.max()))
    FRAG.put("los.gt30_n", int((values > VD.FOLLOWUP_DAYS).sum()))
    FRAG.put("los.primary_field_note",
             "Every discharge definition in this step is derived from the days from "
             "operation to discharge field and never from the billing setting.")

    sameday_n = int((los_observed & los.eq(0)).sum())
    pod1_n = int((los_observed & los.le(1)).sum())
    sameday_clean_n = int((clean & los.eq(0)).sum())
    pod1_clean_n = int((clean & los.le(1)).sum())
    FRAG.put("los.sameday.n", sameday_n)
    FRAG.put("los.sameday.denominator", int(los_observed.sum()))
    FRAG.put("los.sameday.pct", pct(sameday_n, int(los_observed.sum())), 4)
    FRAG.put("los.disch_by_pod1.n", pod1_n)
    FRAG.put("los.disch_by_pod1.denominator", int(los_observed.sum()))
    FRAG.put("los.disch_by_pod1.pct", pct(pod1_n, int(los_observed.sum())), 4)
    FRAG.put("los.sameday.excl.n", sameday_clean_n)
    FRAG.put("los.sameday.excl.denominator", int(clean.sum()))
    FRAG.put("los.sameday.excl.pct", pct(sameday_clean_n, int(clean.sum())), 4)
    FRAG.put("los.disch_by_pod1.excl.n", pod1_clean_n)
    FRAG.put("los.disch_by_pod1.excl.denominator", int(clean.sum()))
    FRAG.put("los.disch_by_pod1.excl.pct", pct(pod1_clean_n, int(clean.sum())), 4)

    expired_dest = frame["DISCHDEST"].astype("string").fillna("").eq("Expired")
    died = frame["MORT30"].fillna(0).astype(int).eq(1)
    still_in = frame["STILLINHOSP"].astype("string").fillna("").str.casefold().eq("yes")
    FRAG.put("los.notdischargedhome.n", int(excluded.sum()))
    FRAG.put("los.notdischargedhome.expired_dest_n", int(expired_dest.sum()))
    FRAG.put("los.notdischargedhome.mort30_n", int(died.sum()))
    FRAG.put("los.notdischargedhome.stillinhosp_n", int(still_in.sum()))
    FRAG.put("los.sameday.notdischargedhome_n",
             int((los_observed & los.eq(0) & excluded).sum()))
    FRAG.put("los.sameday.expired_dest_n", int((los_observed & los.eq(0) & expired_dest).sum()))
    FRAG.put("los.sameday.mort30_n", int((los_observed & los.eq(0) & died).sum()))
    FRAG.put("los.sameday.stillinhosp_n", int((los_observed & los.eq(0) & still_in).sum()))
    FRAG.put("los.disch_by_pod1.notdischargedhome_n",
             int((los_observed & los.le(1) & excluded).sum()))
    FRAG.put("los.disch_by_pod1.expired_dest_n",
             int((los_observed & los.le(1) & expired_dest).sum()))
    FRAG.put("los.disch_by_pod1.mort30_n", int((los_observed & los.le(1) & died).sum()))
    FRAG.put("los.disch_by_pod1.stillinhosp_n",
             int((los_observed & los.le(1) & still_in).sum()))
    FRAG.put("los.exclusion_rule",
             "A patient with an expired discharge destination, a recorded death within 30 "
             "days, or a still in hospital flag did not go home and is excluded from the "
             "discharge-timing rates that carry the exclusion label. Both forms are reported.")

    # -------------------------------------------------------------------
    # TABLE 1
    # -------------------------------------------------------------------
    columns, rows = build_table1(frame)
    write_table(TABLES_DIR / "table1_baseline_by_discharge_timing.csv", columns, rows)
    print(f"Table 1  {len(rows)} rows x {len(columns)} columns")
    strata = _strata(frame)
    FRAG.put("baseline.strata", [label for _, label, _ in strata])
    for key, _, mask in strata:
        FRAG.put(f"baseline.stratum.{key}.n", int(mask.sum()))
        FRAG.put(f"baseline.stratum.{key}.pct", pct(int(mask.sum()), analytic_n), 4)
    FRAG.put("baseline.rows", len(rows))
    FRAG.put("baseline.missing_stratum_handling",
             "The discharge-timing stratum is the empty string for the patients whose "
             "postoperative length of stay is not recorded, and the label layer returns an "
             "empty string for it, so it is routed explicitly to the not-recorded label and "
             "shown as its own column rather than dropped.")
    FRAG.put("baseline.continuous_median_rule",
             "Fusion levels, operative time and postoperative length of stay are reported as "
             "median and interquartile range; the remaining continuous variables are "
             "reported as mean and standard deviation.")
    FRAG.put("baseline.test_rule",
             "Comparisons across the three recorded discharge-timing strata use the "
             "Kruskal-Wallis test for continuous variables and the chi-square test for "
             "categorical and binary variables. The not-recorded stratum is displayed but "
             "not tested.")
    for name in ("AGE", "MALE", "ASA_GE3", "LEVELS_CAT", "BMI"):
        pass  # headline stratum values live in the CSV; the fragment carries stratum sizes

    # -------------------------------------------------------------------
    # TABLE 2 and the trend
    # -------------------------------------------------------------------
    columns, rows, by_year = build_table2(frame, excluded)
    write_table(TABLES_DIR / "table2_los_and_destination_by_year.csv", columns, rows)
    print(f"Table 2  {len(rows)} rows x {len(columns)} columns")

    FRAG.put("trend.years", by_year["years"])
    FRAG.put("trend.patients_by_year", by_year["n"])
    FRAG.put("trend.denominator_by_year", by_year["los_den"])
    FRAG.put("trend.sameday.n_by_year", by_year["sameday_n"])
    FRAG.put("trend.sameday.pct_by_year", by_year["sameday_pct"], 4)
    FRAG.put("trend.disch_by_pod1.n_by_year", by_year["pod1_n"])
    FRAG.put("trend.disch_by_pod1.pct_by_year", by_year["pod1_pct"], 4)
    FRAG.put("trend.excl.denominator_by_year", by_year["clean_den"])
    FRAG.put("trend.excl.sameday.n_by_year", by_year["sameday_clean_n"])
    FRAG.put("trend.excl.sameday.pct_by_year", by_year["sameday_clean_pct"], 4)
    FRAG.put("trend.excl.disch_by_pod1.n_by_year", by_year["pod1_clean_n"])
    FRAG.put("trend.excl.disch_by_pod1.pct_by_year", by_year["pod1_clean_pct"], 4)
    FRAG.put("trend.los.median_by_year", by_year["los_median"])
    FRAG.put("trend.los.q1_by_year", by_year["los_q1"])
    FRAG.put("trend.los.q3_by_year", by_year["los_q3"])
    FRAG.put("trend.covid_year", COVID_YEAR)
    FRAG.put("trend.covid_shown_not_smoothed", True)
    FRAG.put("trend.covid_note",
             "The pandemic year is plotted and annotated rather than smoothed, and the trend "
             "is reported with and without it.")

    year_series = frame["OPERYR"].astype("Int64")
    trend_definitions = (
        ("sameday", los.eq(0), "true same-day discharge"),
        ("disch_by_pod1", los.le(1), "discharge by postoperative day 1"),
    )
    trend_fits: Dict[str, Dict[str, Any]] = {}
    for key, positive, _ in trend_definitions:
        successes, totals, scores = [], [], []
        for value in YEARS:
            mask = los_observed & year_series.eq(value)
            successes.append(int((mask & positive).sum()))
            totals.append(int(mask.sum()))
            scores.append(value)
        full = cochran_armitage(successes, totals, scores)
        keep = [i for i, value in enumerate(YEARS) if value != COVID_YEAR]
        without = cochran_armitage([successes[i] for i in keep],
                                   [totals[i] for i in keep],
                                   [scores[i] for i in keep])
        trend_fits[key] = {"ca_all": full, "ca_no2020": without}
        for label, record in (("all", full), ("no2020", without)):
            prefix = f"trend.ca.{key}.{label}"
            FRAG.put(f"{prefix}.z", record["z"], 4)
            FRAG.put(f"{prefix}.chi2", record["chi2"], 4)
            FRAG.put(f"{prefix}.p", record["p"], 10)
            FRAG.put(f"{prefix}.slope_pp_per_year", record["slope_per_year"], 4)
            # What the FITTED slope implies across the observed window, so nobody has
            # to multiply a slope by a year count in prose. The multiplier is the
            # number of year GAPS, one fewer than the number of years.
            n_gaps = len([v for v in YEARS if label == "all" or v != COVID_YEAR]) - 1
            FRAG.put(f"{prefix}.year_gaps_n", n_gaps)
            FRAG.put(f"{prefix}.fitted_change_pp",
                     record["slope_per_year"] * n_gaps, 4)

        # THE ENDPOINT DIFFERENCE, published as its own key rather than left for `08`
        # to subtract by hand (CLAUDE.md hard rule 3). It is NOT the fitted slope times
        # the window, and pairing the two as though it were is arithmetically
        # incoherent: the series is not required to be monotone, and for discharge by
        # postoperative day 1 it is not. The peak and the monotonicity flag are
        # published beside it so the shape can be stated rather than assumed.
        pct_series = [s / t * 100.0 if t else float("nan")
                      for s, t in zip(successes, totals)]
        peak_idx = max(range(len(pct_series)), key=lambda i: pct_series[i])
        FRAG.put(f"trend.{key}.first_year", int(YEARS[0]))
        FRAG.put(f"trend.{key}.last_year", int(YEARS[-1]))
        FRAG.put(f"trend.{key}.first_year_pct", pct_series[0], 4)
        FRAG.put(f"trend.{key}.last_year_pct", pct_series[-1], 4)
        FRAG.put(f"trend.{key}.endpoint_difference_pp",
                 pct_series[-1] - pct_series[0], 4)
        FRAG.put(f"trend.{key}.peak_year", int(YEARS[peak_idx]))
        FRAG.put(f"trend.{key}.peak_pct", pct_series[peak_idx], 4)
        FRAG.put(f"trend.{key}.peak_is_an_endpoint",
                 bool(peak_idx in (0, len(pct_series) - 1)))
        FRAG.put(f"trend.{key}.monotone_increasing",
                 bool(all(b >= a for a, b in zip(pct_series, pct_series[1:]))))
        FRAG.put(f"trend.{key}.endpoint_difference_note",
                 "The endpoint difference is the last observed year minus the first "
                 "observed year. It is a description of two points, not a fitted "
                 "quantity, and it is not the fitted slope multiplied by the window: "
                 "the two agree only on a perfectly linear series. Report whichever "
                 "one the sentence is actually about, name it as what it is, and do "
                 "not present one as the other.")

    # The adjusted trend: operative year as a LINEAR term, adjusted for age, sex, body mass
    # index, ASA class and fusion levels (PLAN.md 6.3). Only the year term is reported.
    trend_covariates = ["AGE_PER10", "MALE", "BMI_PER5", "ASA_GE3", "LEVELS_CAT"]
    trend_frame = frame.copy()
    trend_frame["YEAR_LINEAR"] = year_series.astype("Float64") - float(YEARS[0])
    complete = los_observed.copy()
    for column in trend_covariates:
        if column == "LEVELS_CAT":
            complete &= trend_frame["LEVELS_CAT"].astype("string").fillna("").ne("")
        else:
            complete &= trend_frame[column].notna()
    complete &= trend_frame["YEAR_LINEAR"].notna()
    for key, positive, _ in trend_definitions:
        subset = trend_frame[complete].copy()
        subset["TREND_OUTCOME"] = positive[complete].astype(int).to_numpy()
        crude = fit_logit(subset, "TREND_OUTCOME", "YEAR_LINEAR", [])
        adjusted = fit_logit(subset, "TREND_OUTCOME", "YEAR_LINEAR", trend_covariates)
        subset_no_covid = subset[subset["OPERYR"].astype("Int64").ne(COVID_YEAR)]
        crude_no_covid = fit_logit(subset_no_covid, "TREND_OUTCOME", "YEAR_LINEAR", [])
        adjusted_no_covid = fit_logit(subset_no_covid, "TREND_OUTCOME", "YEAR_LINEAR",
                                      trend_covariates)
        trend_fits[key].update({"crude": crude, "adjusted": adjusted,
                                "crude_no2020": crude_no_covid,
                                "adjusted_no2020": adjusted_no_covid})
        for label, fit in (("crude", crude), ("adjusted", adjusted),
                           ("crude_no2020", crude_no_covid),
                           ("adjusted_no2020", adjusted_no_covid)):
            prefix = f"trend.logit.{key}.{label}"
            FRAG.put(f"{prefix}.or_per_year", fit["or"], 6)
            FRAG.put(f"{prefix}.lo", fit["lo"], 6)
            FRAG.put(f"{prefix}.hi", fit["hi"], 6)
            FRAG.put(f"{prefix}.p", fit["p"], 10)
            FRAG.put(f"{prefix}.n", fit["n"])
            FRAG.put(f"{prefix}.events", fit["events"])
    FRAG.put("trend.logit.covariates",
             ["Age, per decade", "Male sex", "Body mass index, per 5 units",
              "ASA class 3 or 4", "Fusion levels"])
    FRAG.put("trend.logit.year_term", "operative year as a linear term, per additional year")
    FRAG.put("trend.logit.complete_case_n", int(complete.sum()))
    FRAG.put("trend.logit.dropped_incomplete_n", int(len(frame) - complete.sum()))
    FRAG.put("trend.logit.no_discrimination_statistic", True)
    print("trend      Cochran-Armitage and adjusted logistic fitted for both definitions")

    # -------------------------------------------------------------------
    # Destination by year. NO TREND CLAIM (CLAUDE.md 9.6).
    # -------------------------------------------------------------------
    FRAG.put("dest.years", by_year["years"])
    FRAG.put("dest.denominator_by_year", by_year["dest_den"])
    FRAG.put("dest.home.n_by_year", by_year["home_n"])
    FRAG.put("dest.home.pct_by_year", by_year["home_pct"], 4)
    FRAG.put("dest.facility.n_by_year", by_year["facility_n"])
    FRAG.put("dest.facility.pct_by_year", by_year["facility_pct"], 4)
    FRAG.put("dest.other.n_by_year", by_year["other_n"])
    FRAG.put("dest.other.pct_by_year", by_year["other_pct"], 4)
    FRAG.put("dest.missing_n_by_year", by_year["dest_missing_n"])
    dest_cat = frame["DEST_CAT"].astype("string").fillna("")
    nonhome = frame["NONHOME_DISCH"]
    for label, mask in (("2012_2020", year_series.lt(DEST_ERA_BREAK_YEAR)),
                        ("2021_2024", year_series.ge(DEST_ERA_BREAK_YEAR))):
        den = int((mask & nonhome.notna()).sum())
        num = int((mask & nonhome.fillna(0).astype(int).eq(1)).sum())
        FRAG.put(f"dest.era.{label}.n", int(mask.sum()))
        FRAG.put(f"dest.era.{label}.nonhome_n", num)
        FRAG.put(f"dest.era.{label}.denominator", den)
        FRAG.put(f"dest.era.{label}.nonhome_pct", pct(num, den), 4)
        observed_levels = frame.loc[mask, "DISCHDEST"].astype("string").fillna("")
        FRAG.put(f"dest.era.{label}.destination_levels",
                 int(observed_levels[observed_levels.ne("")].nunique()))
    FRAG.put("dest.era_break_year", DEST_ERA_BREAK_YEAR)
    FRAG.put("dest.trend_claim_permitted", False)
    FRAG.put("dest.note",
             "The discharge destination vocabulary changes completely at the era break, "
             "collapsing from seven facility levels to two, so the non-home discharge rate "
             "is not comparable across the break. The distribution is reported by year "
             "because the plan requires it, and no claim that non-home discharge declined is "
             "made anywhere in this manuscript.")
    FRAG.put("dest.home_definition",
             "Home discharge is taken from the explicit destination level map, which maps "
             "the level meaning a facility that had been the patient's residence to the "
             "facility group. It is never defined by matching the word Home.")
    FRAG.put("dest.home_n", int(dest_cat.eq("Home").sum()))
    FRAG.put("dest.facility_n", int(dest_cat.eq("Facility").sum()))
    FRAG.put("dest.other_n", int(dest_cat.eq("Other").sum()))
    FRAG.put("dest.missing_n", int(dest_cat.eq("").sum()))
    FRAG.put("dest.denominator", int(dest_cat.ne("").sum()))

    # -------------------------------------------------------------------
    # TABLE 4 and 4b: the measurement critique
    # -------------------------------------------------------------------
    outpatient = frame["INOUT"].astype("string").fillna("").eq(VD.INOUT_OUTPATIENT_LEVEL)
    sameday = los.eq(0)
    usable = los_observed & frame["INOUT_OUTPT"].notna()
    stats_all = agreement_stats(outpatient[usable], sameday[usable])
    if stats_all["n"] != int(usable.sum()):
        raise SystemExit("the 2x2 cells do not sum to the stated denominator. Stop condition.")

    columns, rows = build_table4(stats_all)
    write_table(TABLES_DIR / "table4_inout_agreement.csv", columns, rows)
    columns, rows, per_year = build_table4b(frame, outpatient, sameday, usable, stats_all)
    write_table(TABLES_DIR / "table4b_inout_by_year.csv", columns, rows)
    print(f"Table 4  2x2 on {stats_all['n']:,} rows, kappa {stats_all['kappa']:.3f}")

    FRAG.put("inout.denominator", stats_all["n"])
    FRAG.put("inout.excluded_no_los_n", int((~usable).sum()))
    FRAG.put("inout.cell.outpatient_sameday", stats_all["a"])
    FRAG.put("inout.cell.outpatient_night", stats_all["b"])
    FRAG.put("inout.cell.inpatient_sameday", stats_all["c"])
    FRAG.put("inout.cell.inpatient_night", stats_all["d"])
    FRAG.put("inout.outpatient_n", stats_all["outpatient_n"])
    FRAG.put("inout.outpatient_pct", stats_all["outpatient_pct"], 4)
    FRAG.put("inout.inpatient_n", stats_all["inpatient_n"])
    FRAG.put("inout.sameday_n", stats_all["sameday_n"])
    FRAG.put("inout.sameday_pct", stats_all["sameday_pct"], 4)
    FRAG.put("inout.night_n", stats_all["night_n"])
    FRAG.put("inout.outpatient_night_n", stats_all["b"])
    FRAG.put("inout.outpatient_night_pct", stats_all["outpatient_night_pct"], 4)
    for name in ("sensitivity", "specificity", "ppv", "npv", "agreement"):
        FRAG.put(f"inout.{name}", stats_all[name], 4)
        FRAG.put(f"inout.{name}.lo", stats_all[f"{name}_ci"][0], 4)
        FRAG.put(f"inout.{name}.hi", stats_all[f"{name}_ci"][1], 4)
    FRAG.put("inout.disagreement", stats_all["disagreement"], 4)
    FRAG.put("inout.kappa", stats_all["kappa"], 6)
    FRAG.put("inout.kappa.se", stats_all["kappa_se"], 6)
    FRAG.put("inout.kappa.lo", stats_all["kappa_lo"], 6)
    FRAG.put("inout.kappa.hi", stats_all["kappa_hi"], 6)
    FRAG.put("inout.pooled_is_secondary", True)
    FRAG.put("inout.primary_form",
             "The year-by-year statistics are the primary form. The pooled statistic is "
             "reported only beside them, because the raw billing column steps twice inside "
             "the study window and a pooled figure would hide both steps.")
    FRAG.put("inout.is_replication", True)
    FRAG.put("inout.prior_art_pmid", "29351096")
    FRAG.put("inout.replication_note",
             "This measurement critique is a replication and extension of a published "
             "analysis of the same billing field on ACDF in the same registry. It is never "
             "described as new, novel or first. What extends it is the window carried to "
             "2024, the larger measured sample, the year-by-year agreement and kappa, and "
             "the diagnostic-accuracy framing.")
    FRAG.put("inout.years", per_year["years"])
    for key in ("n", "a", "b", "c", "d", "outpatient_n", "sameday_n", "outpatient_night_n"):
        FRAG.put(f"inout.by_year.{key}", per_year[key])
    for key in ("outpatient_pct", "sameday_pct", "outpatient_night_pct", "sensitivity",
                "specificity", "ppv", "npv", "agreement", "disagreement"):
        FRAG.put(f"inout.by_year.{key}", per_year[key], 4)
    FRAG.put("inout.by_year.kappa", per_year["kappa"], 6)
    FRAG.put("inout.by_year.kappa.lo", per_year["kappa_lo"], 6)
    FRAG.put("inout.by_year.kappa.hi", per_year["kappa_hi"], 6)
    kappas = [k for k in per_year["kappa"] if k is not None]
    FRAG.put("inout.by_year.kappa_min", min(kappas), 6)
    FRAG.put("inout.by_year.kappa_max", max(kappas), 6)
    FRAG.put("inout.by_year.kappa_min_year",
             per_year["years"][per_year["kappa"].index(min(kappas))])
    FRAG.put("inout.by_year.kappa_max_year",
             per_year["years"][per_year["kappa"].index(max(kappas))])
    disagreements = [d for d in per_year["disagreement"] if d is not None]
    FRAG.put("inout.by_year.disagreement_min", min(disagreements), 4)
    FRAG.put("inout.by_year.disagreement_max", max(disagreements), 4)
    FRAG.put("inout.by_year.outpatient_night_pct_min", min(per_year["outpatient_night_pct"]), 4)
    FRAG.put("inout.by_year.outpatient_night_pct_max", max(per_year["outpatient_night_pct"]), 4)
    FRAG.put("inout.step_change_years", [2017, 2021])
    FRAG.put("inout.step_change_note",
             "The share of cases carrying the outpatient billing label steps at two points "
             "inside the window, and the later step coincides with the discharge destination "
             "vocabulary change. A change in a billing label is not read as a change in "
             "practice anywhere in this manuscript.")

    # The same agreement analysis restricted to patients who actually went home, as a
    # sensitivity on the target condition rather than on the label.
    clean_usable = usable & ~excluded
    stats_clean = agreement_stats(outpatient[clean_usable], sameday[clean_usable])
    FRAG.put("inout.excl.denominator", stats_clean["n"])
    FRAG.put("inout.excl.kappa", stats_clean["kappa"], 6)
    FRAG.put("inout.excl.agreement", stats_clean["agreement"], 4)
    FRAG.put("inout.excl.sensitivity", stats_clean["sensitivity"], 4)
    FRAG.put("inout.excl.specificity", stats_clean["specificity"], 4)
    FRAG.put("inout.excl.ppv", stats_clean["ppv"], 4)
    FRAG.put("inout.excl.npv", stats_clean["npv"], 4)
    FRAG.put("inout.excl.outpatient_night_pct", stats_clean["outpatient_night_pct"], 4)

    # -------------------------------------------------------------------
    # TABLE 4c: the both-ways comparison
    # -------------------------------------------------------------------
    columns, rows, bothways = build_bothways(frame)
    write_table(TABLES_DIR / "table4c_both_ways_comparison.csv", columns, rows)
    shift = bothways["shift"]
    print(f"Table 4c both-ways on {shift['n']:,} patients, {shift['events']:,} outcomes")

    FRAG.put("bothways.outcome_label", VD.short(BOTHWAYS_OUTCOME))
    FRAG.put("bothways.outcome_prespec_reason", BOTHWAYS_OUTCOME_REASON)
    FRAG.put("bothways.covariates",
             ["Age, per decade", "Male sex", "Body mass index, per 5 units",
              "ASA class 3 or 4", "Fusion levels", "Operative year"])
    FRAG.put("bothways.covariates_identical_between_definitions", True)
    FRAG.put("bothways.no_discrimination_statistic", True)
    FRAG.put("bothways.n", shift["n"])
    FRAG.put("bothways.dropped_incomplete_n", shift["dropped"])
    FRAG.put("bothways.events", shift["events"])
    FRAG.put("bothways.event_pct", pct(shift["events"], shift["n"]), 4)
    for key, _, definition in BOTHWAYS_EXPOSURES:
        record = bothways["results"][key]
        FRAG.put(f"bothways.{key}.definition", definition)
        FRAG.put(f"bothways.{key}.exposed_n", record["exposed_n"])
        FRAG.put(f"bothways.{key}.exposed_events", record["exposed_events"])
        FRAG.put(f"bothways.{key}.exposed_pct",
                 pct(record["exposed_events"], record["exposed_n"]), 4)
        FRAG.put(f"bothways.{key}.unexposed_n", record["unexposed_n"])
        FRAG.put(f"bothways.{key}.unexposed_events", record["unexposed_events"])
        FRAG.put(f"bothways.{key}.unexposed_pct",
                 pct(record["unexposed_events"], record["unexposed_n"]), 4)
        for label, fit in (("crude", record["crude"]), ("adjusted", record["adjusted"])):
            FRAG.put(f"bothways.{key}.{label}.or", fit["or"], 6)
            FRAG.put(f"bothways.{key}.{label}.lo", fit["lo"], 6)
            FRAG.put(f"bothways.{key}.{label}.hi", fit["hi"], 6)
            FRAG.put(f"bothways.{key}.{label}.p", fit["p"], 10)
    FRAG.put("bothways.shift.or_ratio", shift["or_ratio"], 6)
    FRAG.put("bothways.shift.or_difference", shift["or_difference"], 6)
    FRAG.put("bothways.shift.pct_change", shift["pct_change"], 4)
    FRAG.put("bothways.shift.ci_overlap", shift["ci_overlap"])
    FRAG.put("bothways.shift.direction_changed", shift["direction_changed"])
    FRAG.put("bothways.discordant_n", shift["discordant_n"])
    FRAG.put("bothways.discordant_pct", shift["discordant_pct"], 4)
    FRAG.put("bothways.no_difference_test_note",
             "No p-value is computed for the difference between the two estimates: they are "
             "fitted on the same patients and are therefore correlated, so a two-sample "
             "comparison would be invalid.")

    # -------------------------------------------------------------------
    # GATE 2
    # -------------------------------------------------------------------
    gate2: List[Dict[str, Any]] = []
    for year_value, carried_sameday, carried_pod1 in ((2012, 7.1, 68.9), (2024, None, 67.6)):
        mask = los_observed & year_series.eq(year_value)
        den = int(mask.sum())
        sd = int((mask & los.eq(0)).sum())
        p1 = int((mask & los.le(1)).sum())
        med = float(pd.to_numeric(los[mask], errors="coerce").dropna().median())
        FRAG.put(f"los.gate2.sameday.{year_value}.n", sd)
        FRAG.put(f"los.gate2.sameday.{year_value}.denominator", den)
        FRAG.put(f"los.gate2.sameday.{year_value}.pct", pct(sd, den), 4)
        FRAG.put(f"los.gate2.disch_by_pod1.{year_value}.n", p1)
        FRAG.put(f"los.gate2.disch_by_pod1.{year_value}.denominator", den)
        FRAG.put(f"los.gate2.disch_by_pod1.{year_value}.pct", pct(p1, den), 4)
        FRAG.put(f"los.gate2.median.{year_value}", med)
        if carried_sameday is not None:
            FRAG.put(f"los.gate2.sameday.{year_value}.carried_pct", carried_sameday)
            FRAG.put(f"los.gate2.sameday.{year_value}.reproduces",
                     bool(round(pct(sd, den), 1) == carried_sameday))
        FRAG.put(f"los.gate2.disch_by_pod1.{year_value}.carried_pct", carried_pod1)
        FRAG.put(f"los.gate2.disch_by_pod1.{year_value}.reproduces",
                 bool(round(pct(p1, den), 1) == carried_pod1))
        FRAG.put(f"los.gate2.median.{year_value}.carried", 1.0)
        FRAG.put(f"los.gate2.median.{year_value}.reproduces", bool(med == 1.0))
        gate2.append({"year": year_value, "sameday_pct": pct(sd, den),
                      "pod1_pct": pct(p1, den), "median": med, "denominator": den})
    FRAG.put("los.gate2.sameday.2024.resolved_point_estimate",
             pct(int((los_observed & year_series.eq(2024) & los.eq(0)).sum()),
                 int((los_observed & year_series.eq(2024)).sum())), 4)
    FRAG.put("los.gate2.sameday.2024.carried_range_low", 16.0)
    FRAG.put("los.gate2.sameday.2024.carried_range_high", 18.0)
    FRAG.put("los.gate2.cohort_reported", "analytic")
    FRAG.put("los.gate2.note",
             "Every carried-over figure is re-derived on the analytic cohort, which is the "
             "population every other number in this manuscript uses. A mismatch is reported, "
             "never overwritten in either direction.")

    # -------------------------------------------------------------------
    # The carried-over outpatient count, reconciled.
    # -------------------------------------------------------------------
    identified_frame, nonagenarian = load_identified_for_reconciliation()
    gate2_identified = gate2_on_identified(identified_frame, nonagenarian)
    for record in gate2_identified:
        if record["rung"] != "identified":
            continue
        prefix = f"los.gate2.reconcile.gate_{record['gate']}.{record['year']}"
        FRAG.put(f"{prefix}.denominator", record["denominator"])
        FRAG.put(f"{prefix}.sameday_pct", record["sameday_pct"], 4)
        FRAG.put(f"{prefix}.disch_by_pod1_pct", record["pod1_pct"], 4)
        FRAG.put(f"{prefix}.median", record["median"])
    reproduces_2012 = [
        r for r in gate2_identified
        if r["year"] == 2012 and round(r["sameday_pct"], 1) == 7.1
        and round(r["pod1_pct"], 1) == 68.9
    ]
    reproduces_2024 = [
        r for r in gate2_identified
        if r["year"] == 2024 and round(r["pod1_pct"], 1) == 67.6
    ]
    FRAG.put("los.gate2.reconcile.rungs_tested", len(gate2_identified))
    FRAG.put("los.gate2.reconcile.2012_pair_reproduced_n", len(reproduces_2012))
    FRAG.put("los.gate2.reconcile.2012_pair_reproduced_on",
             sorted({r["rung"] for r in reproduces_2012}) or ["none"])
    FRAG.put("los.gate2.reconcile.2024_pod1_reproduced_n", len(reproduces_2024))
    closest_2024 = min((r for r in gate2_identified if r["year"] == 2024),
                       key=lambda r: abs(r["pod1_pct"] - 67.6))
    FRAG.put("los.gate2.reconcile.2024_pod1_closest_pct", closest_2024["pod1_pct"], 4)
    FRAG.put("los.gate2.reconcile.2024_pod1_closest_rung", closest_2024["rung"])
    FRAG.put("los.gate2.reconcile.2024_pod1_closest_delta_pp",
             closest_2024["pod1_pct"] - 67.6, 4)
    FRAG.put("los.gate2.reconcile.note",
             "The carried-over discharge-timing figures were measured on the pre-exclusion "
             "population, not on the analytic cohort. The manuscript reports the analytic "
             "figure with its denominator and states the reconciliation; neither value is "
             "overwritten.")

    reconciliation = reconcile_carryover_outpatient_count(identified_frame, nonagenarian)
    matches = [c for c in reconciliation["candidates"] if c["denominator_match"]]
    pair_matches = [c for c in reconciliation["candidates"] if c["pair_match"]]
    FRAG.put("inout.reconcile.carried_denominator", 47713)
    FRAG.put("inout.reconcile.carried_numerator", 34879)
    FRAG.put("inout.reconcile.carried_pct", 73.0)
    FRAG.put("inout.reconcile.candidates_tested", len(reconciliation["candidates"]))
    FRAG.put("inout.reconcile.denominator_matches", len(matches))
    FRAG.put("inout.reconcile.pair_matches", len(pair_matches))
    FRAG.put("inout.reconcile.analytic_outpatient_n", int(outpatient.sum()))
    FRAG.put("inout.reconcile.analytic_outpatient_los_recorded_n",
             int((outpatient & los_observed).sum()))
    FRAG.put("inout.reconcile.analytic_outpatient_night_n", stats_all["b"])
    FRAG.put("inout.reconcile.analytic_outpatient_night_pct",
             stats_all["outpatient_night_pct"], 4)
    if matches:
        best = matches[0]
        FRAG.put("inout.reconcile.match_gate",
                 "reconciliation ladder" if best["gate"] == "a" else "primary ladder")
        FRAG.put("inout.reconcile.match_rung", best["rung"])
        FRAG.put("inout.reconcile.match_restriction", best["restriction_label"])
        FRAG.put("inout.reconcile.match_denominator", best["denominator"])
        FRAG.put("inout.reconcile.match_numerator", best["numerator"])
        FRAG.put("inout.reconcile.match_pct", best["pct"], 4)
        FRAG.put("inout.reconcile.match_numerator_delta", 34879 - best["numerator"])
    for candidate in reconciliation["candidates"]:
        if candidate["restriction"] != "all":
            continue
        prefix = (f"inout.reconcile.gate_{candidate['gate']}."
                  f"{candidate['rung']}")
        FRAG.put(f"{prefix}.outpatient_n", candidate["denominator"])
        FRAG.put(f"{prefix}.outpatient_night_n", candidate["numerator"])
        FRAG.put(f"{prefix}.outpatient_night_pct", candidate["pct"], 4)
    FRAG.put("inout.reconcile.elective_hypothesis_tested", True)
    elective_b = [c for c in reconciliation["candidates"]
                  if c["gate"] == "b" and c["rung"] == "elective" and c["restriction"] == "all"][0]
    identified_b = [c for c in reconciliation["candidates"]
                    if c["gate"] == "b" and c["rung"] == "identified" and c["restriction"] == "all"][0]
    FRAG.put("inout.reconcile.elective_hypothesis_removes_outpatients",
             identified_b["denominator"] - elective_b["denominator"])
    FRAG.put("inout.reconcile.elective_hypothesis_holds", False)
    FRAG.put("inout.reconcile.reported_cohort", "analytic")
    FRAG.put("inout.reconcile.note",
             "The manuscript reports the billing-label analysis on the analytic cohort. The "
             "pre-exclusion figures exist only so the carried-over count can be reconciled, "
             "and none of them is a denominator for a reported percentage.")

    # -------------------------------------------------------------------
    # Length-of-stay field agreement (the S4 cross-check)
    # -------------------------------------------------------------------
    check = frame["LOS_POSTOP_CHECK"]
    both_present = los_observed & check.notna()
    agree_flag = frame["LOS_POSTOP_AGREE"]
    agree_n = int((both_present & agree_flag.fillna(0).astype(int).eq(1)).sum())
    disagree_mask = both_present & agree_flag.fillna(1).astype(int).eq(0)
    disagree_n = int(disagree_mask.sum())
    deltas = (check[disagree_mask] - los[disagree_mask]).astype(float)
    FRAG.put("los.crosscheck.both_present_n", int(both_present.sum()))
    FRAG.put("los.crosscheck.agree_n", agree_n)
    FRAG.put("los.crosscheck.disagree_n", disagree_n)
    FRAG.put("los.crosscheck.disagree_pct", pct(disagree_n, int(both_present.sum())), 4)
    FRAG.put("los.crosscheck.delta_min", float(deltas.min()) if disagree_n else None)
    FRAG.put("los.crosscheck.delta_max", float(deltas.max()) if disagree_n else None)
    FRAG.put("los.crosscheck.all_one_day_longer",
             bool(disagree_n and float(deltas.min()) == 1.0 and float(deltas.max()) == 1.0))
    FRAG.put("los.crosscheck.parent_disagree_n", 204)
    FRAG.put("los.crosscheck.parent_both_present_n", 112523)
    FRAG.put("los.crosscheck.reproduces_parent",
             bool(disagree_n == 204 and int(both_present.sum()) == 112523))
    sameday_check_n = int((both_present & check.eq(0)).sum())
    pod1_check_n = int((both_present & check.le(1)).sum())
    FRAG.put("los.crosscheck.sameday_n", sameday_check_n)
    FRAG.put("los.crosscheck.sameday_pct", pct(sameday_check_n, int(both_present.sum())), 4)
    FRAG.put("los.crosscheck.disch_by_pod1_n", pod1_check_n)
    FRAG.put("los.crosscheck.disch_by_pod1_pct", pct(pod1_check_n, int(both_present.sum())), 4)

    # -------------------------------------------------------------------
    # S4 sensitivity table
    # -------------------------------------------------------------------
    s4_columns = ["Sensitivity analysis", "Quantity", "Primary analysis",
                  "Sensitivity analysis result", "Difference", "Denominator"]
    s4_rows: List[Dict[str, Any]] = []

    def s4(analysis: str, quantity: str, primary: str, sensitivity: str,
           difference: str, denominator: str) -> None:
        s4_rows.append({"Sensitivity analysis": analysis, "Quantity": quantity,
                        "Primary analysis": primary,
                        "Sensitivity analysis result": sensitivity,
                        "Difference": difference, "Denominator": denominator})

    for key, _, label in trend_definitions:
        record = trend_fits[key]
        s4("Trend excluding the pandemic year",
           f"Cochran-Armitage test for trend in {label}",
           f"z = {record['ca_all']['z']:.2f}, {VD.fmt_p(record['ca_all']['p'])}",
           f"z = {record['ca_no2020']['z']:.2f}, {VD.fmt_p(record['ca_no2020']['p'])}",
           f"{record['ca_no2020']['slope_per_year'] - record['ca_all']['slope_per_year']:+.3f}"
           " percentage points per year",
           VD.fmt_count(int(los_observed.sum())))
        s4("Trend excluding the pandemic year",
           f"Adjusted odds of {label} per additional operative year",
           VD.fmt_or_ci(record["adjusted"]["or"], record["adjusted"]["lo"],
                        record["adjusted"]["hi"], 3),
           VD.fmt_or_ci(record["adjusted_no2020"]["or"], record["adjusted_no2020"]["lo"],
                        record["adjusted_no2020"]["hi"], 3),
           f"{record['adjusted_no2020']['or'] - record['adjusted']['or']:+.4f}",
           VD.fmt_count(record["adjusted"]["n"]))

    s4("Length-of-stay field agreement",
       "Postoperative length of stay, two independent readings",
       "Days from operation to discharge, the direct field",
       "Total hospital stay minus days from admission to operation, the subtraction",
       f"{VD.fmt_count(disagree_n)} rows disagree "
       f"({VD.fmt_pct(pct(disagree_n, int(both_present.sum())), 3)}), every one of them "
       f"exactly one day, the subtraction reading longer",
       VD.fmt_count(int(both_present.sum())))
    s4("Length-of-stay field agreement", "True same-day discharge",
       VD.fmt_n_pct_denom(int((both_present & los.eq(0)).sum()), int(both_present.sum())),
       VD.fmt_n_pct_denom(sameday_check_n, int(both_present.sum())),
       f"{sameday_check_n - int((both_present & los.eq(0)).sum()):+,d} patients",
       VD.fmt_count(int(both_present.sum())))
    s4("Length-of-stay field agreement", "Discharge by postoperative day 1",
       VD.fmt_n_pct_denom(int((both_present & los.le(1)).sum()), int(both_present.sum())),
       VD.fmt_n_pct_denom(pod1_check_n, int(both_present.sum())),
       f"{pod1_check_n - int((both_present & los.le(1)).sum()):+,d} patients",
       VD.fmt_count(int(both_present.sum())))

    s4("Excluding patients who did not go home", "True same-day discharge, all years",
       VD.fmt_n_pct_denom(sameday_n, int(los_observed.sum())),
       VD.fmt_n_pct_denom(sameday_clean_n, int(clean.sum())),
       f"{pct(sameday_clean_n, int(clean.sum())) - pct(sameday_n, int(los_observed.sum())):+.3f}"
       " percentage points",
       VD.fmt_count(int(clean.sum())))
    s4("Excluding patients who did not go home",
       "Discharge by postoperative day 1, all years",
       VD.fmt_n_pct_denom(pod1_n, int(los_observed.sum())),
       VD.fmt_n_pct_denom(pod1_clean_n, int(clean.sum())),
       f"{pct(pod1_clean_n, int(clean.sum())) - pct(pod1_n, int(los_observed.sum())):+.3f}"
       " percentage points",
       VD.fmt_count(int(clean.sum())))
    s4("Excluding patients who did not go home",
       "Agreement between the billing label and the length of stay",
       VD.fmt_pct(stats_all["agreement"]), VD.fmt_pct(stats_clean["agreement"]),
       f"{stats_clean['agreement'] - stats_all['agreement']:+.3f} percentage points",
       VD.fmt_count(stats_clean["n"]))
    s4("Excluding patients who did not go home", "Cohen kappa",
       f"{stats_all['kappa']:.3f}", f"{stats_clean['kappa']:.3f}",
       f"{stats_clean['kappa'] - stats_all['kappa']:+.4f}",
       VD.fmt_count(stats_clean["n"]))

    write_table(TABLES_DIR / "s4_sensitivity_05.csv", s4_columns, s4_rows)
    print(f"S4       {len(s4_rows)} sensitivity rows")

    # -------------------------------------------------------------------
    # Write the fragment and the notes
    # -------------------------------------------------------------------
    # BEFORE the write: every published key must resolve to exactly one declared family
    # (.build_facts.md 9.2 item 4, 12.3). A key added without a family fails here.
    n_families = assert_scale_coverage(FRAG.keys())
    print(f"scale coverage {len(FRAG.keys())} keys, every one matched by exactly one of "
          f"{n_families} declared families")

    n_keys = FRAG.write(FRAGMENT_PATH)
    print(f"fragment {n_keys} keys -> {FRAGMENT_PATH.relative_to(HERE)}")

    write_notes(frame, stats_all, stats_clean, per_year, by_year, trend_fits,
                bothways, reconciliation, gate2, gate2_identified,
                {"disagree_n": disagree_n, "both_present": int(both_present.sum()),
                 "agree_n": agree_n},
                {"sameday_n": sameday_n, "pod1_n": pod1_n,
                 "sameday_clean_n": sameday_clean_n, "pod1_clean_n": pod1_clean_n,
                 "los_observed": int(los_observed.sum()), "clean": int(clean.sum()),
                 "excluded": int(excluded.sum())})

    verify_fragment()
    print()
    print("05_inout.py COMPLETE")
    return 0


def verify_fragment() -> None:
    """Re-read the fragment through the orchestrator's own validator."""
    sys.path.insert(0, str(HERE))
    try:
        import run_all  # noqa: PLC0415  (imported here so a broken orchestrator cannot block analysis)
    except ModuleNotFoundError:
        # run_all.py is the pipeline orchestrator and is not distributed with the
        # public analysis repository. The fragment is written either way; only this
        # namespace self-check is unavailable.
        print("fragment self-validation skipped: run_all.py is not distributed "
              "with the public analysis repository")
        return

    data = run_all.load_fragment(FRAGMENT_PATH)
    violations = run_all.namespace_violations("05", FRAGMENT_PATH, data)
    if violations:
        raise SystemExit(f"namespace violations in results_05.json: {violations}")
    warnings = run_all.key_convention_warnings("results_05.json", data)
    if warnings:
        raise SystemExit("key convention violations:\n" + "\n".join(warnings))
    for key, value in data.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise SystemExit(f"fragment key {key} carries a non-finite value")
    print(f"fragment validated: {len(data)} keys, namespace violations NONE")


# ===========================================================================
# 9. notes_05.md
# ===========================================================================
PRESPEC_SECTION = """## 0. PRE-SPECIFICATION for the both-ways comparison, recorded BEFORE the first fit

Written 2026-07-31, before any regression in this step was run. `PLAN.md` 6.4 requires the
headline "outpatient versus inpatient ACDF" complication comparison to be re-estimated twice
on the same cohort, once with the exposure defined by the billing label and once by true
postoperative length of stay, and says the size of the shift is **to be measured**. No
direction is asserted here.

**Outcome, fixed before the fit: unplanned readmission within 30 days
(`READMIT_UNPLANNED`).** One outcome, chosen a priori, for four stated reasons:

1. It is the outcome the outpatient-spine safety literature leads with, so re-estimating it
   under both exposure definitions speaks directly to the published claims the measurement
   critique targets.
2. It is the most frequent adverse outcome available in this cohort, so the comparison is
   powered to show a shift between two definitions rather than to show sampling noise. The
   rarer outcomes (hematoma requiring reoperation, composite airway morbidity) would give two
   wide, overlapping intervals from which no shift could be read either way.
3. It is ascertained after discharge. An in-hospital complication mechanically lengthens the
   index stay and would therefore help to define the very exposure it is being regressed on,
   building reverse causation into the length-of-stay arm and into that arm only. Readmission
   does not have that defect.
4. It lies entirely outside the companion paper's reserved results. `CLAUDE.md` 7.1 reserves
   the hematoma-stratified consequence cascade and the multivariable hematoma risk model; this
   comparison is not stratified by hematoma status, does not use the hematoma outcome, and
   reports no covariate odds ratio.

**Covariate set, fixed and identical between the two fits**
(`variable_dict.BOTHWAYS_COVARIATES`, `CLAUDE.md` 6.6): age per decade, male sex, body mass
index per 5 units, ASA class 3 or 4, fusion levels (categorical, reference 1 level), operative
year (categorical, reference 2012). Nothing else. Both fits run on **one common complete-case
analysis set**, so the two estimates differ only in how the exposure is defined.

**Effect measure:** odds ratio with a 95% Wald confidence interval, exposure coded 1 for
"outpatient" under each definition so the two are directly comparable. **No discrimination
statistic is computed** (`CLAUDE.md` 6.6). The shift is quantified as the ratio of the two
adjusted odds ratios, their absolute difference, and whether the intervals overlap. **No
p-value is computed for the difference between the two estimates**: they are fitted on the
same patients and are correlated, so a two-sample comparison would be invalid.

This section is reproduced verbatim from the revision of this file that was written before the
first fit, which is what fixes the pre-specification in time.
"""


def write_notes(frame, stats_all, stats_clean, per_year, by_year, trend_fits,
                bothways, reconciliation, gate2, gate2_identified,
                crosscheck, counts) -> None:
    shift = bothways["shift"]
    inout_fit = bothways["results"]["inout"]
    los_fit = bothways["results"]["los"]
    matches = [c for c in reconciliation["candidates"] if c["denominator_match"]]
    pair_matches = [c for c in reconciliation["candidates"] if c["pair_match"]]
    best = matches[0] if matches else None

    def row(candidate: Dict[str, Any]) -> str:
        return (f"| Gate {candidate['gate'].upper()} | {candidate['rung']} | "
                f"{candidate['restriction_label']} | "
                f"{candidate['denominator']:,} | {candidate['numerator']:,} | "
                f"{candidate['pct']:.2f}% |")

    lines: List[str] = []
    lines.append("# notes_05.md -- handoff from `05_inout.py`")
    lines.append("")
    lines.append("Owner: **T7 (length of stay and the `INOUT` critique)**. Owned files: "
                 "`05_inout.py`, `outputs/results_05.json`, `derived/notes_05.md`, "
                 "`tables-csv/table1_baseline_by_discharge_timing.csv`, "
                 "`tables-csv/table2_los_and_destination_by_year.csv`, "
                 "`tables-csv/table4_inout_agreement.csv`, "
                 "`tables-csv/table4b_inout_by_year.csv`, "
                 "`tables-csv/table4c_both_ways_comparison.csv`, "
                 "`tables-csv/s4_sensitivity_05.csv`. Nothing else was created or modified.")
    lines.append("")
    lines.append("**Re-derive, never copy a number out of this file into a table or into "
                 "prose.** Every figure below is a key in `outputs/results_05.json`; read the "
                 "key.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(PRESPEC_SECTION)
    lines.append("---")
    lines.append("")

    # 1
    lines.append("## 1. THE HEADLINE: the billing label against true postoperative length of stay")
    lines.append("")
    lines.append("**This is a REPLICATION, not a new finding.** Bovonratwet et al., *Clin Spine "
                 "Surg* 2018 (PMID 29351096) published this critique on ACDF in this registry, "
                 "using NSQIP 2005-2014, reporting **3,204 of 4,123 (77.7%)** cases the "
                 "registry termed outpatient with a length of stay greater than zero, and "
                 "running the same both-ways multivariable comparison. The words new, novel "
                 "and first must never attach to this finding (`CLAUDE.md` 6a.1). `08` must "
                 "cite PMID 29351096 in the Introduction and the Discussion and state plainly "
                 "that this is a replication.")
    lines.append("")
    lines.append(f"**The 2x2, analytic cohort, denominator {stats_all['n']:,}** "
                 f"(the {len(frame) - stats_all['n']} patients whose postoperative length of "
                 f"stay is not recorded cannot enter a cross-tabulation against it):")
    lines.append("")
    lines.append("| Billing setting | Zero postoperative nights | One or more nights | Total |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Outpatient | {stats_all['a']:,} | {stats_all['b']:,} | "
                 f"{stats_all['outpatient_n']:,} |")
    lines.append(f"| Inpatient | {stats_all['c']:,} | {stats_all['d']:,} | "
                 f"{stats_all['inpatient_n']:,} |")
    lines.append(f"| **Total** | **{stats_all['sameday_n']:,}** | **{stats_all['night_n']:,}** "
                 f"| **{stats_all['n']:,}** |")
    lines.append("")
    lines.append(f"The four cells sum to {stats_all['n']:,} exactly, and sensitivity, "
                 f"specificity and both predictive values are each recomputable from them.")
    lines.append("")
    lines.append("| Statistic | Value | 95% confidence interval | Numerator / denominator |")
    lines.append("|---|---:|---|---|")
    for label, name, num, den in (
            ("Sensitivity", "sensitivity", stats_all["a"], stats_all["sameday_n"]),
            ("Specificity", "specificity", stats_all["d"], stats_all["night_n"]),
            ("Positive predictive value", "ppv", stats_all["a"], stats_all["outpatient_n"]),
            ("Negative predictive value", "npv", stats_all["d"], stats_all["inpatient_n"]),
            ("Raw agreement", "agreement", stats_all["a"] + stats_all["d"], stats_all["n"])):
        ci = stats_all[f"{name}_ci"]
        lines.append(f"| {label} | {stats_all[name]:.1f}% | "
                     f"{ci[0]:.1f} to {ci[1]:.1f} | {num:,} / {den:,} |")
    lines.append(f"| Cohen kappa | {stats_all['kappa']:.3f} | "
                 f"{stats_all['kappa_lo']:.3f} to {stats_all['kappa_hi']:.3f} | "
                 f"{stats_all['n']:,} |")
    lines.append("")
    lines.append(f"**Our comparable figure: {stats_all['b']:,} of "
                 f"{stats_all['outpatient_n']:,} "
                 f"({stats_all['outpatient_night_pct']:.1f}%) cases carrying the outpatient "
                 f"billing label stayed at least one night.** Report it with this denominator "
                 f"and cite Bovonratwet's 3,204 of 4,123 (77.7%) beside it. **Do not present "
                 f"either as confirming the other's exact value** (`CLAUDE.md` 6a.1): the two "
                 f"cohorts overlap and the windows differ.")
    lines.append("")
    lines.append("The sample-size extension is stated as two measured counts and never as an "
                 f"approximate multiple: **{stats_all['outpatient_n']:,} outpatient-labelled "
                 f"ACDF cases here against 4,123 there.**")
    lines.append("")

    # 2
    lines.append("## 2. THE YEAR-BY-YEAR FORM IS PRIMARY. A pooled kappa hides two step changes")
    lines.append("")
    lines.append("T1 measured two step changes in the raw billing column: the outpatient share "
                 "steps 25.9% (2016) to 36.1% (2017), and 44.3% (2020) to 50.6% (2021), the "
                 "later step coinciding exactly with the discharge-destination vocabulary swap "
                 "and the heart-failure prevalence step. **A pooled 2012-2024 kappa would hide "
                 "both**, so the pooled statistic above is reported only beside the table "
                 "below (`.build_facts.md` 7.1, Finding 2, binding).")
    lines.append("")
    lines.append("| Year | Denominator | Outpatient label, % | Zero nights, % | "
                 "Labelled outpatient but stayed a night, % | Sensitivity, % | "
                 "Specificity, % | Positive predictive value, % | Agreement, % | Kappa |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, year_value in enumerate(per_year["years"]):
        lines.append(
            f"| {year_value} | {per_year['n'][index]:,} | "
            f"{per_year['outpatient_pct'][index]:.1f} | "
            f"{per_year['sameday_pct'][index]:.1f} | "
            f"{per_year['outpatient_night_pct'][index]:.1f} | "
            f"{per_year['sensitivity'][index]:.1f} | "
            f"{per_year['specificity'][index]:.1f} | "
            f"{per_year['ppv'][index]:.1f} | "
            f"{per_year['agreement'][index]:.1f} | "
            f"{per_year['kappa'][index]:.3f} |")
    kappas = [k for k in per_year["kappa"] if k is not None]
    lines.append("")
    lines.append(f"**Kappa ranges from {min(kappas):.3f} "
                 f"({per_year['years'][per_year['kappa'].index(min(kappas))]}) to "
                 f"{max(kappas):.3f} "
                 f"({per_year['years'][per_year['kappa'].index(max(kappas))]}) against a "
                 f"pooled {stats_all['kappa']:.3f}.** The share of outpatient-labelled cases "
                 f"that stayed at least one night runs "
                 f"{min(per_year['outpatient_night_pct']):.1f}% to "
                 f"{max(per_year['outpatient_night_pct']):.1f}%.")
    lines.append("")
    lines.append("**Binding on `07` and `08`: never read a change in the billing label as a "
                 "change in practice.** The label's meaning moved; the length of stay is the "
                 "measure that did not.")
    lines.append("")

    # 3
    lines.append("## 3. THE BOTH-WAYS COMPARISON: what the definition change is worth")
    lines.append("")
    lines.append(f"One outcome, pre-specified above: **{VD.short(BOTHWAYS_OUTCOME).lower()}**. "
                 f"One covariate set, identical between the two fits. One common complete-case "
                 f"analysis set of **{shift['n']:,} patients** carrying "
                 f"**{shift['events']:,} outcomes** "
                 f"({100.0 * shift['events'] / shift['n']:.2f}%); "
                 f"{shift['dropped']:,} patients are dropped for an incomplete covariate, "
                 f"exposure or length-of-stay value, and the SAME rows are dropped from both "
                 f"fits so nothing but the definition varies.")
    lines.append("")
    lines.append("| Exposure definition | Exposed, n | Exposed with the outcome | "
                 "Unexposed, n | Unexposed with the outcome | Unadjusted odds ratio | "
                 "Adjusted odds ratio |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for key, record, definition in (("Billing setting", inout_fit,
                                     "outpatient billing setting"),
                                    ("True postoperative length of stay", los_fit,
                                     "zero postoperative nights")):
        lines.append(
            f"| {key} ({definition}) | {record['exposed_n']:,} | "
            f"{record['exposed_events']:,} "
            f"({100.0 * record['exposed_events'] / record['exposed_n']:.2f}%) | "
            f"{record['unexposed_n']:,} | {record['unexposed_events']:,} "
            f"({100.0 * record['unexposed_events'] / record['unexposed_n']:.2f}%) | "
            f"{fmt_or(record['crude'])} | {fmt_or(record['adjusted'])} |")
    lines.append("")
    lines.append(f"**The measured shift.** Adjusted odds ratio "
                 f"{los_fit['adjusted']['or']:.3f} under the length-of-stay definition against "
                 f"{inout_fit['adjusted']['or']:.3f} under the billing definition: a ratio of "
                 f"**{shift['or_ratio']:.2f}**, an absolute difference of "
                 f"{shift['or_difference']:+.3f}, and a "
                 f"{shift['pct_change']:+.1f}% change in the estimate. The two confidence "
                 f"intervals "
                 f"{'OVERLAP' if shift['ci_overlap'] else 'DO NOT OVERLAP'}, and the direction "
                 f"of the association relative to the null "
                 f"{'CHANGES' if shift['direction_changed'] else 'does not change'} between "
                 f"the two definitions.")
    lines.append("")
    lines.append(f"**{shift['discordant_n']:,} patients "
                 f"({shift['discordant_pct']:.1f}% of the analysis set) are classified into "
                 f"different exposure groups by the two definitions.** That discordance is the "
                 f"mechanism behind the shift and belongs in the Results beside it.")
    lines.append("")
    lines.append("**No p-value is computed for the difference between the two estimates.** "
                 "They are fitted on the same patients and are correlated, so a two-sample "
                 "comparison would be invalid. No discrimination statistic is computed and no "
                 "covariate odds ratio is reported: the model exists to adjust one exposure "
                 "contrast, not to describe risk (`CLAUDE.md` 6.6).")
    lines.append("")

    # 4
    lines.append("## 4. Discharge timing 2012 to 2024, and the trend tested two ways")
    lines.append("")
    lines.append("Every definition is on the direct days-from-operation-to-discharge field and "
                 "never on the billing label.")
    lines.append("")
    lines.append("| Year | Length of stay recorded | True same-day discharge, % | "
                 "Discharge by postoperative day 1, % | Postoperative stay, median (IQR) | "
                 "Home, % | Facility, % |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for index, year_value in enumerate(by_year["years"]):
        mark = " (pandemic year)" if year_value == COVID_YEAR else ""
        lines.append(
            f"| {year_value}{mark} | {by_year['los_den'][index]:,} | "
            f"{by_year['sameday_pct'][index]:.2f} | {by_year['pod1_pct'][index]:.2f} | "
            f"{by_year['los_median'][index]:.0f} ({by_year['los_q1'][index]:.0f} to "
            f"{by_year['los_q3'][index]:.0f}) | {by_year['home_pct'][index]:.2f} | "
            f"{by_year['facility_pct'][index]:.2f} |")
    lines.append("")
    lines.append("**The shape.** True same-day discharge rises from "
                 f"{by_year['sameday_pct'][0]:.2f}% to "
                 f"{by_year['sameday_pct'][-1]:.2f}%, while discharge by postoperative day 1 "
                 f"moves from {by_year['pod1_pct'][0]:.2f}% to "
                 f"{by_year['pod1_pct'][-1]:.2f}% and the median postoperative stay is "
                 f"{by_year['los_median'][0]:.0f} day in every year of the window. The growth "
                 f"in same-day discharge came out of the one-night group, not out of the "
                 f"long-stay tail.")
    lines.append("")
    lines.append("**Both tests, as `PLAN.md` 6.3 requires:**")
    lines.append("")
    lines.append("| Definition | Cochran-Armitage | Trend, percentage points per year | "
                 "Unadjusted odds per year | Adjusted odds per year | Analysis n |")
    lines.append("|---|---|---:|---|---|---:|")
    for key, label in (("sameday", "True same-day discharge"),
                       ("disch_by_pod1", "Discharge by postoperative day 1")):
        record = trend_fits[key]
        lines.append(f"| {label} | z = {record['ca_all']['z']:.2f}, "
                     f"{VD.fmt_p(record['ca_all']['p'])} | "
                     f"{record['ca_all']['slope_per_year']:+.3f} | "
                     f"{VD.fmt_or_ci(record['crude']['or'], record['crude']['lo'], record['crude']['hi'], 3)} | "
                     f"{VD.fmt_or_ci(record['adjusted']['or'], record['adjusted']['lo'], record['adjusted']['hi'], 3)} | "
                     f"{record['adjusted']['n']:,} |")
    lines.append("")
    lines.append("The adjusted model carries operative year as a **linear** term and adjusts "
                 "for age, sex, body mass index, ASA class and fusion levels. Only the year "
                 "term is reported. **Case mix moved over thirteen years and an unadjusted "
                 "trend alone would be challenged**, which is why both are published.")
    lines.append("")
    lines.append("**Read the two rows together and do not read the second as a real change.** "
                 "Same-day discharge rises "
                 f"{FRAG['trend.ca.sameday.all.slope_pp_per_year']:+.2f} percentage points "
                 f"per year on the fitted linear trend. Discharge by postoperative day 1 "
                 f"rises "
                 f"{FRAG['trend.ca.disch_by_pod1.all.slope_pp_per_year']:+.2f} percentage "
                 f"points per year on the same fit, and reaches significance only because "
                 f"the trend test runs on {counts['los_observed']:,} patients.")
    lines.append("")
    lines.append("**Two different quantities, and they must not be paired as though they "
                 "were one.** An earlier revision of this note put the fitted slope and the "
                 "endpoint difference in the same sentence, which does not add up and would "
                 "have reached the manuscript. Both numbers are right; only the pairing was "
                 "wrong. Stated separately, for discharge by postoperative day 1:")
    lines.append("")
    lines.append("| Quantity | What it is | Value |")
    lines.append("|---|---|---:|")
    lines.append(f"| Fitted linear slope | Cochran-Armitage, percentage points per year | "
                 f"{FRAG['trend.ca.disch_by_pod1.all.slope_pp_per_year']:+.4f} |")
    lines.append(f"| Fitted change across the window | that slope over the "
                 f"{FRAG['trend.ca.disch_by_pod1.all.year_gaps_n']} year gaps between "
                 f"{FRAG['trend.disch_by_pod1.first_year']} and "
                 f"{FRAG['trend.disch_by_pod1.last_year']} | "
                 f"{FRAG['trend.ca.disch_by_pod1.all.fitted_change_pp']:+.4f} |")
    lines.append(f"| Endpoint difference | {FRAG['trend.disch_by_pod1.last_year']} minus "
                 f"{FRAG['trend.disch_by_pod1.first_year']}, two observed points | "
                 f"{FRAG['trend.disch_by_pod1.endpoint_difference_pp']:+.4f} |")
    lines.append("")
    lines.append("**The two disagree because the series is not monotone.** Discharge by "
                 f"postoperative day 1 peaks at "
                 f"{FRAG['trend.disch_by_pod1.peak_pct']:.2f}% in "
                 f"{FRAG['trend.disch_by_pod1.peak_year']} and falls back to "
                 f"{FRAG['trend.disch_by_pod1.last_year_pct']:.2f}% by "
                 f"{FRAG['trend.disch_by_pod1.last_year']}, so a fitted straight line and a "
                 f"first-to-last subtraction are answering different questions. Every one of "
                 "these numbers is a fragment key "
                 "(`trend.ca.disch_by_pod1.all.slope_pp_per_year`, `.fitted_change_pp`, "
                 "`trend.disch_by_pod1.endpoint_difference_pp`, `.peak_pct`, `.peak_year`), "
                 "so `08` renders them and types none of them.")
    lines.append("")
    lines.append("**The substantive point is unchanged and must survive any rewrite:** "
                 "discharge by postoperative day 1 is statistically significant and "
                 "clinically negligible. `08` must print the effect size beside the p-value "
                 "for this row, never the p-value alone, and must name each quantity as "
                 "what it is rather than calling either one the trend.")
    lines.append("")
    lines.append(f"**The pandemic year is shown, not smoothed.** {COVID_YEAR} carries "
                 f"{by_year['sameday_pct'][by_year['years'].index(COVID_YEAR)]:.2f}% same-day "
                 f"discharge against "
                 f"{by_year['sameday_pct'][by_year['years'].index(COVID_YEAR) - 1]:.2f}% the "
                 f"year before. `07` must plot it and annotate it. The trend excluding it is "
                 f"in `tables-csv/s4_sensitivity_05.csv` and moves the adjusted estimate for "
                 f"same-day discharge from "
                 f"{trend_fits['sameday']['adjusted']['or']:.4f} to "
                 f"{trend_fits['sameday']['adjusted_no2020']['or']:.4f} per year.")
    lines.append("")

    # 5
    lines.append("## 5. Two things the discharge-timing rates would otherwise get wrong")
    lines.append("")
    lines.append("### 5.1 The same-day flags contain patients who did not go home")
    lines.append("")
    lines.append(f"`.build_facts.md` 7.5 item 8 is binding and is re-derived here. "
                 f"{counts['excluded']:,} analytic patients carry an expired discharge "
                 f"destination, a recorded death within 30 days, or a still-in-hospital flag. "
                 f"Of them, {counts['sameday_n'] - counts['sameday_clean_n']} fall inside the "
                 f"same-day flag and "
                 f"{counts['pod1_n'] - counts['pod1_clean_n']} inside the "
                 f"discharge-by-day-1 flag. **A death on postoperative day 0 is not a "
                 f"discharge.** Both rates are therefore reported twice, and the exclusion is "
                 f"stated rather than applied silently:")
    lines.append("")
    lines.append("| Rate | Including patients who did not go home | Excluding them |")
    lines.append("|---|---|---|")
    lines.append(f"| True same-day discharge | "
                 f"{counts['sameday_n']:,} / {counts['los_observed']:,} "
                 f"({100.0 * counts['sameday_n'] / counts['los_observed']:.2f}%) | "
                 f"{counts['sameday_clean_n']:,} / {counts['clean']:,} "
                 f"({100.0 * counts['sameday_clean_n'] / counts['clean']:.2f}%) |")
    lines.append(f"| Discharge by postoperative day 1 | "
                 f"{counts['pod1_n']:,} / {counts['los_observed']:,} "
                 f"({100.0 * counts['pod1_n'] / counts['los_observed']:.2f}%) | "
                 f"{counts['pod1_clean_n']:,} / {counts['clean']:,} "
                 f"({100.0 * counts['pod1_clean_n'] / counts['clean']:.2f}%) |")
    lines.append("")
    lines.append("The effect is small because the contaminated group is small, and that is "
                 "itself the finding: the exclusion is required for honesty rather than "
                 "because it moves the answer.")
    lines.append("")
    lines.append(f"**Denominator convention, so two nearly equal figures are not mistaken for "
                 f"a discrepancy.** Every rate in this step uses the population in which the "
                 f"quantity is defined, so the same-day rate is "
                 f"{counts['sameday_n']:,} of the {counts['los_observed']:,} patients whose "
                 f"postoperative length of stay is recorded, not of all "
                 f"{len(frame):,}. `02` published the same count over the full analytic "
                 f"cohort as a column-build figure. The counts are identical; only the "
                 f"denominator differs, and this step's is the one a rate should carry.")
    lines.append("")
    lines.append("### 5.2 Discharge destination carries NO trend claim")
    lines.append("")
    lines.append("`CLAUDE.md` 9.6 is binding. The destination distribution is reported by year "
                 "because `PLAN.md` 6.3 requires it, and **no claim that non-home discharge "
                 "declined appears anywhere.** The vocabulary swaps completely at the era "
                 "break and the facility levels collapse from seven to two, so the rate is not "
                 "comparable across it. Home discharge is defined by the explicit level map, "
                 "never by matching the word Home, which would fold a facility level into the "
                 "home group.")
    lines.append("")

    # 6
    lines.append("## 6. GATE 2: every carried-over number, re-derived")
    lines.append("")
    lines.append("`CLAUDE.md` 8.5: a mismatch beyond tolerance is a stop condition to report, "
                 "never a number to overwrite in either direction. The manuscript reports the "
                 "**analytic** cohort (`CLAUDE.md` 4.4).")
    lines.append("")
    lines.append("| Carried-over figure | Analytic cohort, re-derived | Verdict |")
    lines.append("|---|---|---|")
    g2012 = gate2[0]
    g2024 = gate2[1]
    lines.append(f"| Same-day discharge 7.1% in 2012 | {g2012['sameday_pct']:.2f}% "
                 f"({g2012['denominator']:,}) | "
                 f"{'MATCH' if round(g2012['sameday_pct'], 1) == 7.1 else 'MISMATCH'} |")
    lines.append(f"| Same-day discharge in 2024, carried as a 16 to 18% range | "
                 f"**{g2024['sameday_pct']:.2f}%** ({g2024['denominator']:,}) | "
                 f"RESOLVED to a point estimate |")
    lines.append(f"| Home within 1 day 68.9% in 2012 | {g2012['pod1_pct']:.2f}% "
                 f"({g2012['denominator']:,}) | "
                 f"{'MATCH' if round(g2012['pod1_pct'], 1) == 68.9 else 'MISMATCH'} |")
    lines.append(f"| Home within 1 day 67.6% in 2024 | {g2024['pod1_pct']:.2f}% "
                 f"({g2024['denominator']:,}) | "
                 f"{'MATCH' if round(g2024['pod1_pct'], 1) == 67.6 else 'MISMATCH'} |")
    lines.append(f"| Median postoperative stay 1 day in both years | "
                 f"{g2012['median']:.0f} day and {g2024['median']:.0f} day | "
                 f"{'MATCH' if g2012['median'] == 1 and g2024['median'] == 1 else 'MISMATCH'} |")
    lines.append("")
    lines.append("**The three mismatches have one cause, and it is the same cause `02` found "
                 "for the carried-over post-discharge fraction: the carried-over figures are "
                 "PRE-EXCLUSION numbers.** Measured on the pre-exclusion frame, which is where "
                 "`02` also had to go to reproduce the 41.3%, the same two definitions land "
                 "here:")
    lines.append("")
    lines.append("| Ladder | Rung | Year | Denominator | Same-day, % | By postoperative day "
                 "1, % | Median stay |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for record in gate2_identified:
        if record["rung"] != "identified":
            continue
        lines.append(f"| Gate {record['gate'].upper()} | {record['rung']} | "
                     f"{record['year']} | {record['denominator']:,} | "
                     f"{record['sameday_pct']:.3f} | {record['pod1_pct']:.3f} | "
                     f"{record['median']:.0f} |")
    lines.append("")
    ident_2012 = [r for r in gate2_identified
                  if r["rung"] == "identified" and r["year"] == 2012 and r["gate"] == "b"][0]
    ident_2024 = [r for r in gate2_identified
                  if r["rung"] == "identified" and r["year"] == 2024 and r["gate"] == "b"][0]
    reproduces_2024 = [r for r in gate2_identified
                       if r["year"] == 2024 and round(r["pod1_pct"], 1) == 67.6]
    lines.append(f"**The 2012 pair reproduces exactly on the pre-exclusion cohort:** "
                 f"{ident_2012['sameday_pct']:.3f}% rounds to 7.1% and "
                 f"{ident_2012['pod1_pct']:.3f}% rounds to 68.9%, both to one decimal, which "
                 f"is what identifies the denominator behind the carried-over figures.")
    lines.append("")
    if reproduces_2024:
        lines.append("The 2024 discharge-by-day-1 figure also reproduces there.")
    else:
        closest = min((r for r in gate2_identified if r["year"] == 2024),
                      key=lambda r: abs(r["pod1_pct"] - 67.6))
        lines.append(f"**The 2024 discharge-by-day-1 figure does NOT reproduce exactly "
                     f"anywhere, on any rung of either ladder.** The closest is the "
                     f"pre-exclusion rung at {closest['pod1_pct']:.3f}%, which rounds to "
                     f"{closest['pod1_pct']:.1f}% against the carried-over 67.6%, a gap of "
                     f"{closest['pod1_pct'] - 67.6:+.3f} percentage points. It is consistent "
                     f"with the carried-over value having been truncated rather than rounded "
                     f"at one decimal, but that is an inference and it is not asserted. "
                     f"**Reported, not reconciled away** (`CLAUDE.md` 8.5).")
    lines.append("")
    lines.append("On the analytic cohort, which is the population every other number in this "
                 "manuscript uses, all three measures are higher, because the excluded cases "
                 "are disproportionately the longer-staying ones. **Report the analytic figure "
                 "with its denominator and state the reconciliation. Do not adopt the "
                 "carried-over value and do not discard it.**")
    lines.append("")
    lines.append(f"**The 2024 same-day rate is resolved: {g2024['sameday_pct']:.2f}% of "
                 f"{g2024['denominator']:,} on the analytic cohort.** `PLAN.md` 13's open item "
                 f"is closed. The carried-over 16 to 18% range brackets both the analytic "
                 f"{g2024['sameday_pct']:.2f}% and the pre-exclusion "
                 f"{ident_2024['sameday_pct']:.2f}%, which is the most likely reason it was "
                 f"carried as a range rather than a point.")
    lines.append("")

    # 7
    lines.append("## 7. The carried-over 47,713, RESOLVED")
    lines.append("")
    lines.append("`PLAN.md` 13 and `PLAN.md` 7.5 leave the denominator behind "
                 "\"34,879 of 47,713 (73%)\" unrecorded. T1 measured 47,749 on the Gate B "
                 "pre-exclusion cohort and 47,735 on Gate A; `02` measured 46,046 on the "
                 "analytic cohort. None is 47,713. T1 recorded one explicitly unverified "
                 "hypothesis for this step to test first and did not adopt it.")
    lines.append("")
    lines.append("**Result 1: T1's elective hypothesis is REFUTED.** The elective rung removes "
                 f"{[c for c in reconciliation['candidates'] if c['gate'] == 'b' and c['rung'] == 'identified' and c['restriction'] == 'all'][0]['denominator'] - [c for c in reconciliation['candidates'] if c['gate'] == 'b' and c['rung'] == 'elective' and c['restriction'] == 'all'][0]['denominator']:,} "
                 "outpatient-labelled cases, not the 36 that would have been needed. The "
                 "elective rung lands on "
                 f"{[c for c in reconciliation['candidates'] if c['gate'] == 'b' and c['rung'] == 'elective' and c['restriction'] == 'all'][0]['denominator']:,}, "
                 "nowhere near the carried figure.")
    lines.append("")
    lines.append(f"**Result 2: 47,713 reproduces EXACTLY, and on exactly one of the "
                 f"{len(reconciliation['candidates'])} candidate populations tested.**")
    lines.append("")
    lines.append("| Ladder | Rung | Restriction | Outpatient-labelled | Stayed a night | Share |")
    lines.append("|---|---|---|---:|---:|---:|")
    for candidate in reconciliation["candidates"]:
        if candidate["restriction"] == "all" or candidate["denominator_match"]:
            lines.append(row(candidate))
    lines.append("")
    if best is not None:
        lines.append(f"**The carried-over 47,713 is the Gate {best['gate'].upper()} "
                     f"pre-exclusion cohort, outpatient billing label, restricted to the rows "
                     f"where the cross-check length-of-stay field is computable** (that is, "
                     f"where the total hospital stay and the days from admission to operation "
                     f"are both recorded). That population carries exactly "
                     f"{best['denominator']:,} cases.")
        lines.append("")
        lines.append(f"**The numerator does NOT reproduce exactly.** On that same population "
                     f"{best['numerator']:,} cases stayed at least one night, "
                     f"{34879 - best['numerator']} short of the carried-over 34,879. The share "
                     f"is {best['pct']:.2f}% either way, so the carried-over \"73%\" "
                     f"reproduces to one decimal as {best['pct']:.1f}%. The residual "
                     f"{abs(34879 - best['numerator'])}-case gap is "
                     f"{100.0 * abs(34879 - best['numerator']) / 34879:.3f}% of the numerator "
                     f"and is reported, not reconciled away (`CLAUDE.md` 8.5).")
    if not pair_matches:
        lines.append("")
        lines.append("**No tested population reproduces the carried-over PAIR exactly.**")
    lines.append("")
    lines.append(f"**What the manuscript reports** (`CLAUDE.md` 4.4): the analytic cohort. "
                 f"{stats_all['b']:,} of {stats_all['outpatient_n']:,} "
                 f"({stats_all['outpatient_night_pct']:.1f}%) outpatient-labelled cases stayed "
                 f"at least one night, out of {int((frame['INOUT'] == VD.INOUT_OUTPATIENT_LEVEL).sum()):,} "
                 f"carrying the label in total. The pre-exclusion figures above are "
                 f"reconciliation only and are never a denominator for a reported percentage.")
    lines.append("")

    # 8
    lines.append("## 8. Sensitivity analyses owned by this step")
    lines.append("")
    lines.append("Both are in `tables-csv/s4_sensitivity_05.csv`.")
    lines.append("")
    lines.append("**The trend excluding the pandemic year.** Reported for both definitions and "
                 "both tests. The direction and the significance of the trend are unchanged; "
                 "the sizes move as the table records.")
    lines.append("")
    lines.append(f"**The two length-of-stay readings against each other.** "
                 f"{crosscheck['both_present']:,} analytic rows carry both, "
                 f"{crosscheck['agree_n']:,} agree and **{crosscheck['disagree_n']} disagree "
                 f"({100.0 * crosscheck['disagree_n'] / crosscheck['both_present']:.3f}%), "
                 f"every one of them exactly one day, the subtraction reading longer.** "
                 f"`CLAUDE.md` 8.7 records the companion project's measurement as 204 "
                 f"disagreements on 112,523 rows: "
                 f"{'REPRODUCED exactly' if crosscheck['disagree_n'] == 204 and crosscheck['both_present'] == 112523 else 'NOT reproduced'}. "
                 f"The direct field stays the analysis variable; the subtraction is never one.")
    lines.append("")

    # 9
    lines.append("## 9. What `07` and `08` must know")
    lines.append("")
    lines.append("1. **Read `outputs/results.json`, never this file.** Every number above is a "
                 "key under `los.`, `trend.`, `dest.`, `inout.`, `bothways.` or `baseline.`. "
                 "This project already has recorded instances of a number copied out of a "
                 "handoff note being wrong while the artefact was right.")
    lines.append("2. **Figure 4** (discharge timing by year) is built from "
                 "`trend.years`, `trend.sameday.pct_by_year`, "
                 "`trend.disch_by_pod1.pct_by_year` and `trend.los.median_by_year`. "
                 f"**Mark {COVID_YEAR} and do not smooth it.**")
    lines.append("3. **Figure 5** (the billing label against the length of stay) is built from "
                 "the four `inout.cell.*` keys for the agreement panel and "
                 "`inout.by_year.disagreement` for the year-by-year panel. The year-by-year "
                 "panel is the primary one.")
    lines.append("4. **The `INOUT` critique is a replication.** The Introduction and the "
                 "Discussion must cite PMID 29351096 and say so. The words new, novel, first "
                 "and unrecognised must never attach to it. What may be claimed as an "
                 "extension: the window carried to 2024, the larger sample stated as two "
                 "measured counts rather than a multiple, the year-by-year agreement and "
                 "kappa, and the diagnostic-accuracy framing.")
    lines.append("5. **Never read a change in the billing label as a change in practice.** The "
                 "label steps twice inside the window.")
    lines.append("6. **No claim that non-home discharge declined.** Report the destination "
                 "distribution by year and state the era artefact in Limitations.")
    lines.append("7. **The same-day and by-day-1 rates are reported with the exclusion of "
                 "patients who did not go home stated explicitly**, and both forms are "
                 "published so the effect of the exclusion is visible.")
    lines.append("8. **The carried-over discharge-timing figures are pre-exclusion numbers.** "
                 "The Methods should say so once, in the same sentence that states the "
                 "analytic denominator, exactly as `04` will for the post-discharge fraction.")
    lines.append("9. **The both-ways comparison reports one outcome, pre-specified before the "
                 "fit**, one covariate set, one analysis set, and no discrimination "
                 "statistic. Its point is the definition change and nothing else varies.")
    lines.append("10. **Table 1 has a fourth column** for the patients whose discharge timing "
                 "is not recorded. It is displayed rather than dropped, and it is not "
                 "included in any test.")
    lines.append("11. **THIS STEP'S `.lo` AND `.hi` KEYS CARRY THREE DIFFERENT UNITS.** "
                 "`inout.agreement.lo` and the four diagnostic-accuracy families are "
                 "**percentages**; `inout.kappa.lo` and `inout.by_year.kappa.lo` are a "
                 "**kappa** on the minus 1 to 1 scale; `bothways.*.lo` and "
                 "`trend.logit.*.lo` are **odds ratios**. Nothing in a key name "
                 "distinguishes them, `03_timing.py` publishes the same suffixes as "
                 "proportions and `06_window.py` publishes them as percentages, so a "
                 "scale rule from another step must never be applied here. Each of this "
                 "step's six namespaces now carries its own declaration (`los.scale`, "
                 "`trend.scale`, `dest.scale`, `inout.scale`, `bothways.scale`, "
                 "`baseline.scale`), with a machine-readable family list and unit list "
                 "beside it (`.build_facts.md` section 9).")
    lines.append("12. **Do not pair the fitted trend slope with the endpoint difference.** "
                 "They are different quantities and on a non-monotone series they do not "
                 "agree: discharge by postoperative day 1 has a fitted slope of "
                 f"{FRAG['trend.ca.disch_by_pod1.all.slope_pp_per_year']:+.4f} percentage "
                 f"points per year, which is "
                 f"{FRAG['trend.ca.disch_by_pod1.all.fitted_change_pp']:+.4f} across the "
                 f"window, while the endpoint difference is "
                 f"{FRAG['trend.disch_by_pod1.endpoint_difference_pp']:+.4f}. Both are "
                 "published (`trend.ca.<definition>.all.slope_pp_per_year`, "
                 "`.fitted_change_pp`, `trend.<definition>.endpoint_difference_pp`); "
                 "name whichever is used as what it is. See section 4.")
    lines.append("")
    lines.append("## 10. One deviation to be ruled on by the orchestrator")
    lines.append("")
    lines.append("`CLAUDE.md` 6.6 says \"the one regression this project does fit is the "
                 "both-ways comparison\". `PLAN.md` 6.3 separately and explicitly requires the "
                 "discharge-timing trend to be tested with **a logistic model carrying "
                 "operative year as a linear term, adjusted for age, sex, body mass index, ASA "
                 "class and fusion levels**, and to report it beside the Cochran-Armitage "
                 "test. Both are fitted here, because the plan wins on science and because "
                 "6.6's stated purpose is to bar a new hematoma risk model, a points score and "
                 "a discrimination statistic, none of which this step produces. The trend "
                 "model reports its year term only, no covariate odds ratio, and no "
                 "discrimination statistic. Flagged rather than assumed.")
    lines.append("")

    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"notes    {len(lines)} lines -> {NOTES_PATH.relative_to(HERE)}")


if __name__ == "__main__":
    raise SystemExit(main())
