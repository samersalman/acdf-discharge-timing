#!/usr/bin/env python3
"""02_build_cohort.py -- recode the export bundle and freeze the analytic cohort.

PORTED from `projects/ACDF-hematoma/v1/02_build_cohort.py` and EXTENDED for the
discharge-timing project (PLAN.md 11.1). The cohort ladder logic is UNTOUCHED; everything
added here is a new derived column or a new measurement.

This step is the correctness gate of the whole project. Everything downstream reads what
it writes and NEVER re-derives a cohort (CLAUDE.md 4.1).

Inputs
------
    exports/acdf-discharge-timing/cohort.csv  124,344 x 143, from 01_export_cohort.py
    cpt_acdf.py                               CPT / ICD taxonomy and the pinned ladders
    variable_dict.py                          LABELS, SEED, TIMED_OUTCOMES, level maps

Outputs (default mode only)
---------------------------
    derived/cohort_analytic.parquet      the analytic cohort, n = 112,642
    derived/cohort_identified.parquet    pre-exclusion, n = 124,344, for Figure 1
    derived/cohort_schema.md             every derived column, dtype, origin and definition
    derived/notes_02.md                  handoff for 03 / 04 / 05 / 06
    outputs/results_02.json              flat dotted keys (CLAUDE.md 4.2)

Three gates, all of which must pass (CLAUDE.md 8.2, 8.3)
-------------------------------------------------------
    Gate B   default mode, PRIMARY.  124,344 -> 121,210 -> 113,469 -> 112,642
    Gate A   --replicate-plan, RECONCILIATION.  124,266 -> 121,142 -> 113,411 -> 112,585
    Gate 1   adding INOUT to the variable list must not move a single row count.

The event ladder is 642 -> 613 -> 548 -> 541 on BOTH ladders under the CURRENT primary
outcome definition (`cpt_acdf.HEMATOMA_DEFINITION`, V2): the 78 restored nonagenarians
carry zero hematoma events. PLAN.md prints the superseded V0 ladder 640 / 611 / 544 / 537
for the reason CLAUDE.md section 0 and .build_facts.md section 1 document. Every count is
asserted against cpt_acdf.COHORT_LADDER_N / COHORT_LADDER_N_PLAN / COHORT_LADDER_EVENTS;
none is retyped here, so importing lands on the current definition automatically.

Default mode computes BOTH ladders (they differ only in the age cast) so results_02.json
carries the reconciliation as well as the primary. `--replicate-plan` runs the Gate A
ladder as its primary path, asserts it, prints it, and DELIBERATELY WRITES NOTHING, so the
frozen artifacts can never be clobbered by the reconciliation mode.

What this port DELIBERATELY DROPS from the parent, and why
---------------------------------------------------------
  * The parent's GATE D block (its own plan's three subgroup denominators and the
    BMI-window identifiability sweep) and its `burden_variant` block. Both target figures
    printed in the PARENT's governing plan, which this project never quotes, and the burden
    block publishes coagulopathy-burden rates that CLAUDE.md 7.1 forbids this manuscript in
    every form. The BMI plausibility window itself is UNCHANGED, so `BMI`, `BMI_PER5` and
    `MODEL_COMPLETE` stay byte-identical to the parent's and the two cohorts cannot fork.
  * The parent's `cascade_mort.*` block (hematoma versus no-hematoma mortality) and its
    dysphagia counts: both are the parent's results, forbidden here by CLAUDE.md 7.1. The
    COLUMNS are still built, because dropping them would fork the two cohorts.
  * The parent's seroma-guard reconstruction, which re-implements cpt_acdf internals that
    that module's own self-test already covers.

The traps this file is written around (each fails SILENTLY; CLAUDE.md section 9)
------------------------------------------------------------------------------
  * `PRSEPIS` stores the literal string 'None' for "no sepsis". The bundle is read with
    `keep_default_na=False` or the sepsis filter inverts.
  * The elective predicate needs BOTH halves, NULL-safely: `EMERGNCY` stops at 2020,
    `CASETYPE` starts at 2021.
  * The OTHERCPT exclusion scan is NULL-safe by construction (`any_code_matches`); the
    naive SQL form collapses the cohort to 485.
  * The hematoma union is the UNGATED union of all four ICD columns, slot 1 only. A
    year-gated rule loses a real 2016 event ascertained by the ICD-9 arm alone.
  * There is no `DEATH` column. 30-day mortality is DOPERTOD-based (CLAUDE.md 9.4);
    YRDEATH is NOT window-bounded and goes into a separate sensitivity column.
  * REINTUB / FAILWEAN / OUPNEUMO are `!= 'No Complication'`, never `== 'Yes'`.
  * `DISCHDEST` swaps vocabulary completely at 2021 and needs an explicit level map
    covering both; `Facility Which was Home` is NOT counted as home. The map lives in
    `variable_dict.py` here (T2 owns it) and an unmapped level RAISES.
  * Postoperative LOS is `DOPTODIS`, NEVER `INOUT`. `INOUT` is exported in this project
    because the measurement critique needs it, and confusing the two is the error this
    paper documents.
  * Operative year comes from the warehouse year, delivered in the bundle as `PUFYEAR`.
  * `FNSTATUS1` is dead; `FNSTATUS2` is the functional-status column.

Usage
-----
    python3 02_build_cohort.py                  # build and freeze the analytic cohort
    python3 02_build_cohort.py --replicate-plan # Gate A reconciliation; writes nothing
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import cpt_acdf as CPT           # noqa: E402  taxonomy: the only place a code is a literal
import variable_dict as VD       # noqa: E402  labels, SEED, contract constants

np.random.seed(VD.SEED)          # CLAUDE.md hard rule 4 (nothing here draws, but seed anyway)

EXPORT_CSV = PROJECT_DIR / "exports" / "acdf-discharge-timing" / "cohort.csv"
DERIVED_DIR = PROJECT_DIR / "derived"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
ANALYTIC_PARQUET = DERIVED_DIR / "cohort_analytic.parquet"
IDENTIFIED_PARQUET = DERIVED_DIR / "cohort_identified.parquet"
SCHEMA_MD = DERIVED_DIR / "cohort_schema.md"
NOTES_MD = DERIVED_DIR / "notes_02.md"
RESULTS_JSON = OUTPUTS_DIR / "results_02.json"

# The parent's bundle is 142 columns. This project's variable list is the parent's plus
# exactly `INOUT` (.build_facts.md section 4), so the bundle is 143. T1 proved at the export
# layer that deleting `INOUT` from every row of this bundle reproduces the parent's bundle
# byte for byte with an identical SHA-256 (notes_cpt.md 6.3). Gate 1 here is the ladder
# landing on the pinned constants PLUS a structural proof that nothing on the ladder path
# reads the added column, taken by rebuilding both ladders from the raw CSV with the column
# deleted before any derivation runs (see `gate_one`).
EXPECTED_BUNDLE_COLS = 143
PARENT_BUNDLE_COLS = 142
GATE1_ADDED_COLUMNS = ("INOUT",)
MIN_AGE = 18

# NSQIP top-codes age at 90: the raw PUF token is the literal '90+' and the curated view
# ships it as SMALLINT 90. Verified read-only 2026-07-30 on the identified cohort:
# AGE = 90 in exactly 78 rows and the raw token '90' (without the plus) does not occur, so
# AGE == 90 in the bundle is EXACTLY the set the plan's TRY_CAST(AGE AS INT) nulls.
TOPCODED_AGE = 90

# ---------------------------------------------------------------------------
# Decisions this file had to MAKE rather than inherit. Each is stated here once,
# repeated in cohort_schema.md, and reported in derived/notes_02.md.
# ---------------------------------------------------------------------------
# BMI plausibility window, kg/m^2, inclusive. Values outside are set to NaN. It is a
# CHOSEN clinical-plausibility rule, never a discovered one: the handful of cases below 12
# and above 70 are not physiologically credible for an elective, outpatient-heavy ACDF
# population.
#
# PORTED UNCHANGED AND IT MUST STAY UNCHANGED. It is what makes `BMI`, `BMI_PER5` and
# `MODEL_COMPLETE` identical in the two projects; moving a bound here forks the two
# cohorts. The parent's Gate D block, which measured how far this window is identifiable
# from figures printed in the PARENT's plan, is not ported: those figures belong to the
# parent manuscript and this project never quotes them.
BMI_MIN, BMI_MAX = 12.0, 70.0

# Fusion-level cap for the companion per-level term. LEVELS_N is uncapped and its right tail
# runs past 3; the categorical stratifier is 1 / 2 / >= 3. BOTH terms are materialized so the
# two cohorts carry the same columns. THIS PROJECT FITS NEITHER: it fits no hematoma risk
# model at all (CLAUDE.md 6.6).
LEVELS_CAP = 3

# Preoperative anemia, WHO hematocrit thresholds. Sex-specific, so it is undefined where the
# recorded sex is neither male nor female.
ANEMIA_HCT_MALE, ANEMIA_HCT_FEMALE = 39.0, 36.0

# Coagulopathy-burden component thresholds. PORTED; the columns exist so the two cohorts
# cannot fork, and NOTHING in this project ever analyses them (CLAUDE.md 7.1).
INR_THRESHOLD, PLT_THRESHOLD, PTT_THRESHOLD = 1.2, 150.0, 35.0

ASA_KEEP = ("1-No Disturb", "2-Mild Disturb", "3-Severe Disturb", "4-Life Threat")
ASA_MORIBUND = "5-Moribund"
ASA_UNASSIGNED = "None assigned"
NO_SEPSIS = "None"
NO_COMPLICATION = "No Complication"

# `DISCHDEST` swaps vocabulary completely at 2021 (CLAUDE.md 9.5). The parent carried its
# own `DISCHDEST_MAP` literal here; in this project the level map is owned by T2 and lives
# in `variable_dict.py` as `DISCHDEST_LEVEL_MAP`, verified programmatically identical to the
# parent's INCLUDING KEY ORDER (notes_variable_dict.md section 3). This file must NOT carry
# a second copy (notes_variable_dict.md 7.3). Behaviour is preserved exactly:
# `dischdest_category` returns `DEST_CAT_MISSING` (the empty string) for a blank, which is
# what the parent's `.fillna("")` produced, and RAISES on any unmapped non-blank level.
# `Facility Which was Home` is a FACILITY: the ACS level means the patient went to a
# facility that had been their residence before surgery, which is not a discharge to the
# community, and substring-matching "Home" would silently fold it into the home group.

# ---------------------------------------------------------------------------
# THE TIMING-FIELD HANDLING RULE (CLAUDE.md 8.6, PLAN.md 6.1). Applied ONCE, here.
#
# Every days-to-event field in NSQIP is an integer count of postoperative days on a fixed
# 30-day follow-up window. A value is USABLE if and only if it is an integer in
# [TIMING_DAY_MIN, TIMING_DAY_MAX]. Anything else, meaning non-numeric, non-integer,
# negative, or greater than 30, is COUNTED, PUBLISHED per field under `timingfield.*`, and
# set to MISSING in the derived `POD_*` column.
#
# It is never clipped to a boundary and never silently dropped. The affected patient stays
# in the cohort and stays an event; only that patient's TIMING is treated as unobserved, so
# the case appears in row 1 of the PLAN.md 6.2 denominator ladder and drops out at row 2.
# Every downstream step inherits the rule through the frozen `POD_*` columns and NO step
# clips or drops locally.
#
# The rule applies to DAYS-TO-EVENT fields only. It does NOT apply to the length-of-stay
# fields, which are not window-bounded quantities: a patient may legitimately stay in
# hospital for more than 30 days, and truncating a length of stay would corrupt the
# discharge-timing analysis this project exists to do.
# ---------------------------------------------------------------------------
TIMING_DAY_MIN, TIMING_DAY_MAX = 0, 30
TIMING_RANGE_RULE = (
    "A days-to-event value is used only when it is an integer in the range 0 to 30 "
    "postoperative days. Any other value is counted, published with its field, and set to "
    "missing in the derived postoperative-day column. It is never clipped to a boundary and "
    "never silently dropped: the patient stays in the cohort and stays an event, and only "
    "the timing is treated as unobserved."
)

# Every days-to-event field in the bundle, audited within the analytic cohort.
TIMING_FIELDS = (
    "RETORPODAYS", "RETOR2PODAYS", "READMPODAYS1", "READMPODAYS2", "READMPODAYS3",
    "READMPODAYS4", "READMPODAYS5", "DREINTUB", "DFAILWEAN", "DOUPNEUMO", "DOPERTOD",
)
# The length-of-stay family, audited the same way but NOT range-limited.
LOS_FIELDS = ("DOPTODIS", "TOTHLOS", "HTOODAY")

# The strict gate for each raw days-to-event field: the flag whose positivity is supposed to
# be necessary and sufficient for the field to be populated. A day present where the strict
# gate is negative is a data-quality finding.
STRICT_GATE = OrderedDict((
    ("RETORPODAYS", "REOP_ANY"),
    ("DREINTUB", "REINTUBATION"),
    ("DFAILWEAN", "VENT48"),
    ("DOUPNEUMO", "PNEUMONIA"),
    ("READMPODAYS1", "READMIT_UNPLANNED"),
    ("DOPERTOD", "MORT30"),
))

# Roster outcomes whose days-to-event field is legitimately populated on rows where the
# outcome flag is negative, with the documented reason. Counted and reported either way.
NEGFLAG_EXPECTED = {
    "hematoma": "the reoperation day is recorded for every unplanned reoperation, and most "
                "reoperations are not hematoma evacuations",
    "readmit": "the readmission day is recorded for planned readmissions as well as "
               "unplanned ones",
    "airway": "the composite reads the reoperation day for the tracheostomy arm, and that "
              "field is recorded for every unplanned reoperation",
}

# Reoperation slot 2. Column names, not codes.
REOP_SLOT2_CPT_COLUMN = "REOPOR2CPT1"
REOP_SLOT2_PODAYS_COLUMN = "RETOR2PODAYS"

# Figures the constitution recorded, held here so a change shows up as a REPORTED mismatch
# rather than a silent adjustment (CLAUDE.md 8.5). The rung difference itself is computed
# from cpt_acdf and never retyped.
CONSTITUTION_EXCLUSION_BOX_SUM = 828        # CLAUDE.md 9.16
CONSTITUTION_LOS_DISAGREEMENTS = 204        # CLAUDE.md 8.7, measured by the parent
CONSTITUTION_LOS_BOTH_PRESENT = 112523      # CLAUDE.md 8.7

# The nine namespaces `run_all.py` reserves for this fragment (CLAUDE.md 4.2 plus the
# `prespec.` addition ratified in notes_run_all.md section 4). Validated at write time, so a
# namespace violation fails here rather than at the merge.
RESERVED_PREFIXES = ("cohort.", "ladder.", "gate.", "derived.", "timingfield.",
                     "losfield.", "inoutfield.", "dq.", "prespec.")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")

# `RACE_NEW` carries 22 level strings in this cohort, including comma-joined multi-race
# combinations that appear only from 2021. Collapse to the six locked RACE_CAT levels.
# `Race combinations with low frequency` is a single literal level with NO comma and must
# not be treated as a combination string.
RACE_MAP = {
    "White": "White",
    "Black or African American": "Black",
    "Asian": "Asian",
    "American Indian or Alaska Native": "Other",
    "Native Hawaiian or Pacific Islander": "Other",
    "Native Hawaiian or Other Pacific Islander": "Other",
    "Some Other Race": "Other",
    "Race combinations with low frequency": "Other",
    "Middle Eastern or North African": "Other",
    "Hispanic or Latino": "Hispanic",
    "Unknown/Not Reported": "Unknown",
}

SPEC_MAP = {"Neurosurgery": "Neurosurgery", "Orthopedics": "Orthopaedic"}

# Raw NSQIP Yes/No columns recoded IN PLACE to nullable 0/1. The rule, asserted below: a
# raw column is recoded only when variable_dict.FEATURE_PRETTY declares it a model feature
# and it has no derived twin. Everything else keeps its NSQIP text and gets a derived twin.
BINARY_INPLACE = ("BLEEDDIS", "SMOKE", "HXCOPD", "HXCHF", "HYPERMED", "DIALYSIS", "STEROID")

# LABELS entries this step added to variable_dict.py under the controlled add-only
# exception, because each is a derived column created here that had no label. Nothing
# existing was changed. Asserted present at import so the two files cannot drift.
LABELS_ADDED = ("EXCL_ASA_UNASSIGNED", "STEP_ELECTIVE", "STEP_EXCLUSIONS", "STEP_ANALYTIC",
                "MORT_ANY_RECORDED", "LEVELS_PER_LEVEL_CAP3")

NUMERIC_COLS = (
    "AGE", "HEIGHT", "WEIGHT", "PRHCT", "PRALBUM", "PRINR", "PRPLATE", "PRPTT",
    "OPTIME", "WORKRVU", "RETORPODAYS", "RETOR2PODAYS", "READMPODAYS1", "READMPODAYS2",
    "READMPODAYS3", "READMPODAYS4", "READMPODAYS5", "YRDEATH", "DOPERTOD", "DOPTODIS",
    "HTOODAY", "TOTHLOS", "PUFYEAR", "OPERYR",
    # NEW here: the three airway / pneumonia days-to-event fields. The parent left them as
    # passthrough text because it timed no outcome; this project times all eight.
    "DREINTUB", "DFAILWEAN", "DOUPNEUMO",
)

LADDER_ORDER = CPT.COHORT_LADDER_ORDER      # identified, elective, exclusions, analytic

# --- import-time contract checks -------------------------------------------
_missing_labels = [k for k in LABELS_ADDED if k not in VD.LABELS]
if _missing_labels:
    raise SystemExit(
        f"[02] variable_dict.LABELS is missing {_missing_labels}, which 02 creates and "
        f"added. Re-apply the add-only edit and re-run `python3 variable_dict.py`."
    )
# The in-place binary recode rule, asserted rather than asserted-in-prose: recode a raw
# NSQIP column only when FEATURE_PRETTY declares it a model feature.
_not_features = [c for c in BINARY_INPLACE if c not in VD.FEATURE_PRETTY]
if _not_features:
    raise SystemExit(
        f"[02] {_not_features} are recoded in place but are not model features. The rule "
        f"is: recode a raw column only when variable_dict.FEATURE_PRETTY names it."
    )
# notes_variable_dict.md 7.1, BINDING: `PRIMARY_MODEL_TERMS` is deleted in this project
# because it fits no hematoma model. `MODEL_COMPLETE_TERMS` carries the same six columns in
# the same order, so `MODEL_COMPLETE` reproduces byte for byte.
if hasattr(VD, "PRIMARY_MODEL_TERMS"):
    raise SystemExit(
        "[02] variable_dict.PRIMARY_MODEL_TERMS exists again. It was deleted deliberately "
        "because this project fits no hematoma risk model (CLAUDE.md 6.6). Read "
        "MODEL_COMPLETE_TERMS, which is a ported completeness roster and not a model."
    )
_missing_roster = [c for c in VD.MODEL_COMPLETE_TERMS if c not in VD.LABELS]
if _missing_roster:
    raise SystemExit(f"[02] MODEL_COMPLETE_TERMS names unlabelled columns {_missing_roster}")
# The eight timed outcomes are a contract between T2 and T4; neither may invent a name the
# other does not have (.build_facts.md section 5).
for _o in VD.TIMED_OUTCOMES:
    for _name in (_o.event_col, _o.pod_col, VD.postdisch_column(_o.key),
                  VD.timed_both_column(_o.key)) + tuple(_o.raw_fields):
        if _name not in VD.LABELS:
            raise SystemExit(
                f"[02] the timed-outcome roster names {_name!r}, which has no "
                f"variable_dict.LABELS entry. Re-run `python3 variable_dict.py`."
            )
if "POD_HEMATOMA" in VD.LABELS:
    raise SystemExit(
        "[02] variable_dict.LABELS gained POD_HEMATOMA. The roster points the hematoma "
        "outcome at the parent's HEMATOMA_POD column and a second day column must not be "
        "created (notes_variable_dict.md 7.4)."
    )


# ===========================================================================
# UNITS AND SCALES (.build_facts.md section 9)
# ===========================================================================
# The rule is: NEVER infer a scale from a key suffix, and read the declaration published by
# the step that OWNS the key. Section 9.3 closed the fourteen namespaces owned by `03`, `05`
# and `06`; section 9.6 records that this step's nine were never asked for theirs, so a gate
# obeying 9.2 item 4 would fail on every key this file publishes. These declarations close
# that gap for `02`.
#
# Most of what this step publishes is a COUNT, which is unambiguous. It is declared anyway,
# because the point of the rule is that a gate can assert a unit for EVERY key without a
# special case, and because a handful of keys here carry a unit their name does not:
#
#   * the body-mass-index plausibility window bounds, which are not days and not counts;
#   * the death-day extremes, the length-of-stay extremes and the out-of-window day lists,
#     which are days rather than counts;
#   * the count of patients whose postoperative stay was zero days, whose name reads like a
#     number of days and is a number of PATIENTS;
#   * the median postoperative day of reoperation on each rung, published under a bare name
#     with no unit in it at all;
#   * the era-break drop in non-home discharge, which is a RELATIVE percentage of the
#     earlier era's rate and NOT a difference in percentage points;
#   * the pre-specified risk thresholds, held as PROPORTIONS on the 0 to 1 scale beside a
#     list of rendered percentage labels.
#
# `<namespace>.scale` is prose a human can read. `<namespace>.scale.families` and
# `<namespace>.scale.units` carry the same content as two parallel lists so
# `verify_numbers.py` can assert a unit mechanically rather than parse English, and
# `<namespace>.scale.other_numeric_keys` is the catch-all. The family entries are KEY
# PATTERNS addressed to `08`, not rendered prose. No key is renamed and none is removed:
# every value in this fragment has already been verified against the names as they stand.
# ---------------------------------------------------------------------------
UNIT_PATIENTS = "count of patients"
UNIT_EVENTS = "count of events"
UNIT_HEM_EVENTS = "count of hematoma events"
UNIT_PERCENT = "percentage on the 0 to 100 scale"
UNIT_PROPORTION = "proportion on the 0 to 1 scale"
UNIT_POD = "integer postoperative day"
UNIT_DAYS = "days"
UNIT_YEAR = "calendar year"
UNIT_BMI = "body mass index in kilograms per square metre"
UNIT_FUSION_LEVELS = "count of fusion levels"
UNIT_DISTINCT = "count of distinct observed values"
UNIT_VALUES = "count of recorded values, summed across fields"
UNIT_CHECKS = "count of ladder comparisons"
UNIT_COLUMNS = "count of columns"
UNIT_REPLICATES = "count of resampling replicates"
UNIT_SEED = "dimensionless integer, a seed and not a measurement"
UNIT_BOOL = "boolean, carries no unit"
UNIT_TEXT = "text, carries no unit"

SCALE_SCOPE_NOTE = (
    "This declaration governs only the namespace it is named for. It is not a project-wide "
    "rule and must never be read as one: the same est, lo and hi suffixes carry proportions, "
    "percentages, odds ratios and a kappa in the analysis steps, so read the declaration "
    "published by the step that owns the key."
)

SCALE_CATCH_ALL = (
    "Every other numeric key in this namespace is a count of patients or events, a "
    "denominator, or an integer day."
)

# (namespace, prose, ((key pattern, unit), ...))
SCALE_DECLARATIONS = (
    (
        "cohort",
        "Within the cohort-ladder namespace a rung count is a count of PATIENTS, a rung "
        "event count is a count of HEMATOMA events under the primary code set, and the rate "
        "beside them is a percentage on the 0 to 100 scale, being the events over the "
        "patients on that same rung. The exclusion boxes and the flow counts are counts of "
        "patients; the overlap count is the number of patients meeting more than one final "
        "exclusion criterion and is a count, not a difference. The two superseded code sets "
        "publish their analytic event counts here as counts of hematoma events.",
        (
            ("cohort.<rung>.n", UNIT_PATIENTS),
            ("cohort.<rung>.events", UNIT_HEM_EVENTS),
            ("cohort.<rung>.rate_pct", UNIT_PERCENT),
            ("cohort.analytic.events_v0", UNIT_HEM_EVENTS),
            ("cohort.analytic.events_v1", UNIT_HEM_EVENTS),
            ("cohort.analytic.by_year.<year>.n", UNIT_PATIENTS),
            ("cohort.analytic.by_year.<year>.events", UNIT_HEM_EVENTS),
            ("cohort.nonagenarian_delta.<rung>", UNIT_PATIENTS),
            ("cohort.flow.<box>.n", UNIT_PATIENTS),
            ("cohort.flow.exclusion_delta", UNIT_PATIENTS),
            ("cohort.flow.exclusion_overlap_n", UNIT_PATIENTS),
            ("cohort.flow.exclusion_boxes_sum", UNIT_PATIENTS),
            ("cohort.flow.exclusion_boxes_sum_recorded", UNIT_PATIENTS),
            ("cohort.flow.exclusion_boxes_disjoint", UNIT_BOOL),
            ("cohort.flow.exclusion_arithmetic_reproduces", UNIT_BOOL),
            ("cohort.flow.exclusion_arithmetic_note", UNIT_TEXT),
            ("cohort.hematoma.definition", UNIT_TEXT),
            ("cohort.hematoma.definition_label", UNIT_TEXT),
        ),
    ),
    (
        "ladder",
        "The outcome-variant namespace publishes ONE numeric family and it is a count of "
        "hematoma events on a cohort rung under one of the three code sets. There is no "
        "patient count and no percentage in this namespace, so a rate must never be read "
        "from it; the matching patient counts live on the cohort ladder.",
        (
            ("ladder.variant.<version>.<rung>.events", UNIT_HEM_EVENTS),
            ("ladder.variant.<version>.is_primary", UNIT_BOOL),
            ("ladder.variant.<version>.label", UNIT_TEXT),
        ),
    ),
    (
        "gate",
        "Within the gate namespace the reconciliation ladder carries the same three families "
        "as the primary ladder: a count of PATIENTS, a count of HEMATOMA events, and a "
        "percentage on the 0 to 100 scale. The rungs-checked count is a number of "
        "comparisons, not of patients. The column-width keys are counts of COLUMNS in the "
        "export bundle, and the rows-moved key is a count of patients, which is zero by "
        "design. Every verdict key is a boolean.",
        (
            ("gate.a.<rung>.n", UNIT_PATIENTS),
            ("gate.a.<rung>.events", UNIT_HEM_EVENTS),
            ("gate.a.<rung>.rate_pct", UNIT_PERCENT),
            ("gate.a.pass", UNIT_BOOL),
            ("gate.b.pass", UNIT_BOOL),
            ("gate.a.rungs_checked", UNIT_CHECKS),
            ("gate.b.rungs_checked", UNIT_CHECKS),
            ("gate.a.note", UNIT_TEXT),
            ("gate.b.note", UNIT_TEXT),
            ("gate.1.rebuild.<leg>.<rung>.n", UNIT_PATIENTS),
            ("gate.1.rebuild.<leg>.<rung>.events", UNIT_HEM_EVENTS),
            ("gate.1.rebuild.added_columns_present", UNIT_BOOL),
            ("gate.1.rebuild.from_raw_csv", UNIT_BOOL),
            ("gate.1.columns", UNIT_COLUMNS),
            ("gate.1.parent_columns", UNIT_COLUMNS),
            ("gate.1.added_columns", UNIT_TEXT),
            ("gate.1.rows_moved", UNIT_PATIENTS),
            ("gate.1.pass", UNIT_BOOL),
            ("gate.1.ladder_unchanged", UNIT_BOOL),
            ("gate.1.structural_pass", UNIT_BOOL),
            ("gate.1.structural_postderive_pass", UNIT_BOOL),
            ("gate.1.structural_rebuild_pass", UNIT_BOOL),
            ("gate.1.width_pass", UNIT_BOOL),
            ("gate.1.detail", UNIT_TEXT),
            ("gate.1.note", UNIT_TEXT),
            ("gate.1.export_proof", UNIT_TEXT),
            ("gate.1.structural_postderive_detail", UNIT_TEXT),
            ("gate.1.structural_rebuild_detail", UNIT_TEXT),
        ),
    ),
    (
        "derived",
        "The derived-column namespace is mostly counts of patients, and three families in it "
        "are NOT. The body-mass-index window bounds are a body mass index in kilograms per "
        "square metre. The death-day extremes are integer postoperative days, and the "
        "out-of-window day lists and the longest recorded stay among them are DAYS of "
        "postoperative stay, which are not bounded at day 30. The fusion-level cap and "
        "maximum are counts of levels operated on, not counts of patients. Every key ending "
        "in pct is a percentage on the 0 to 100 scale, and so is the completeness share "
        "spelled without that suffix. ONE key is a trap: the era-break drop in non-home "
        "discharge is a RELATIVE percentage, that is the fall expressed as a percentage of "
        "the earlier era's own rate, and it is NOT a difference in percentage points.",
        (
            ("derived.bmi.window_min", UNIT_BMI),
            ("derived.bmi.window_max", UNIT_BMI),
            ("derived.bmi.implausible_analytic_events", UNIT_HEM_EVENTS),
            ("derived.bmi.implausible_analytic_n", UNIT_PATIENTS),
            ("derived.bmi.missing_analytic_n", UNIT_PATIENTS),
            ("derived.levels.cap", UNIT_FUSION_LEVELS),
            ("derived.levels.max", UNIT_FUSION_LEVELS),
            ("derived.levels.above_cap_n", UNIT_PATIENTS),
            ("derived.mort30.dopertod_min", UNIT_POD),
            ("derived.mort30.dopertod_max", UNIT_POD),
            ("derived.mort30.rate_pct", UNIT_PERCENT),
            ("derived.mort_any.rate_pct", UNIT_PERCENT),
            ("derived.mort_any.yrdeath_only_max_los_postop", UNIT_DAYS),
            ("derived.mort_any.out_of_window_los_postop_days", UNIT_DAYS),
            ("derived.mort_any.out_of_window_expired_days", UNIT_DAYS),
            ("derived.mort_any.out_of_window_discharged_alive_days", UNIT_DAYS),
            ("derived.nonhome.era_break_relative_drop_pct", UNIT_PERCENT),
            ("derived.nonhome.<era>.rate_pct", UNIT_PERCENT),
            ("derived.nonhome.<era>.dischdest_levels", UNIT_DISTINCT),
            ("derived.nonhome.<era>.facility_levels", UNIT_DISTINCT),
            ("derived.nonhome.<era>.cases", UNIT_PATIENTS),
            ("derived.nonhome.<era>.denominator", UNIT_PATIENTS),
            ("derived.nonhome.<era>.n", UNIT_PATIENTS),
            ("derived.nonhome.no_trend_claim", UNIT_BOOL),
            ("derived.rate_by_year.min_pct", UNIT_PERCENT),
            ("derived.rate_by_year.max_pct", UNIT_PERCENT),
            ("derived.subgroup.model_complete.pct", UNIT_PERCENT),
            ("derived.subgroup.model_complete.events", UNIT_HEM_EVENTS),
            ("derived.subgroup.model_complete.n", UNIT_PATIENTS),
            ("derived.subgroup.inr_measured.n", UNIT_PATIENTS),
            ("derived.subgroup.lab_complete.n", UNIT_PATIENTS),
            ("derived.subgroup.note", UNIT_TEXT),
            ("derived.window_comp.rate_pct", UNIT_PERCENT),
            ("derived.window_comp.n", UNIT_PATIENTS),
            ("derived.window_comp.hematoma_n", UNIT_PATIENTS),
            ("derived.window_comp.overlap_hematoma_and_airway_n", UNIT_PATIENTS),
            ("derived.window_comp.pod_present_n", UNIT_EVENTS),
            ("derived.window_comp.pod_missing_n", UNIT_EVENTS),
            ("derived.window_comp.components", UNIT_TEXT),
            ("derived.airway.composite_n", UNIT_PATIENTS),
            ("derived.airway.reintubation_n", UNIT_PATIENTS),
            ("derived.airway.vent48_n", UNIT_PATIENTS),
            ("derived.pneumonia.n", UNIT_PATIENTS),
            ("derived.readmit_unplanned.n", UNIT_PATIENTS),
            ("derived.reop_any.n", UNIT_PATIENTS),
            ("derived.reop_trach.slot1_n", UNIT_PATIENTS),
            ("derived.mort30.n", UNIT_PATIENTS),
            ("derived.mort30.dopertod_outside_window_n", UNIT_PATIENTS),
            ("derived.mort_any.n", UNIT_PATIENTS),
            ("derived.mort_any.yrdeath_n", UNIT_PATIENTS),
            ("derived.mort_any.yrdeath_only_n", UNIT_PATIENTS),
            ("derived.mort_any.yrdeath_only_out_of_window_n", UNIT_PATIENTS),
            ("derived.mort_any.yrdeath_only_out_of_window_expired_n", UNIT_PATIENTS),
            ("derived.dischdest.levels_observed_n", UNIT_DISTINCT),
            ("derived.dischdest.home_n", UNIT_PATIENTS),
            ("derived.dischdest.facility_n", UNIT_PATIENTS),
            ("derived.dischdest.facility_which_was_home_n", UNIT_PATIENTS),
            ("derived.dischdest.other_n", UNIT_PATIENTS),
            ("derived.dischdest.missing_n", UNIT_PATIENTS),
            ("derived.era.icd10_n", UNIT_PATIENTS),
            ("derived.era.emergncy_n", UNIT_PATIENTS),
            ("derived.era.icd10_first_year", UNIT_YEAR),
            ("derived.era.emergncy_last_year", UNIT_YEAR),
            ("derived.icd9_only.events_identified", UNIT_HEM_EVENTS),
            ("derived.icd9_only.events_2016", UNIT_HEM_EVENTS),
            ("derived.icd9_only.last_year", UNIT_YEAR),
            ("derived.optime.missing_n", UNIT_PATIENTS),
            ("derived.sex.nonbinary_analytic_n", UNIT_PATIENTS),
            ("derived.slot_union.analytic.slot1_events", UNIT_HEM_EVENTS),
            ("derived.slot_union.analytic.slots_1_2_events", UNIT_HEM_EVENTS),
            ("derived.slot_union.analytic.slot2_added_events", UNIT_HEM_EVENTS),
            ("derived.slot_union.convention", UNIT_TEXT),
            ("derived.slot_union.note", UNIT_TEXT),
            ("derived.trach.invariant_rows_differing_analytic", UNIT_PATIENTS),
            ("derived.trach.invariant_rows_differing_identified", UNIT_PATIENTS),
            ("derived.trach.slot1_analytic_n", UNIT_PATIENTS),
            ("derived.trach.slot12_analytic_n", UNIT_PATIENTS),
            ("derived.trach.slot1_identified_n", UNIT_PATIENTS),
            ("derived.trach.slot12_identified_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_analytic_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_identified_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_already_in_composite_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_analytic_already_in_composite_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_via_reintubation_n", UNIT_PATIENTS),
            ("derived.trach.slot2_only_via_vent48_n", UNIT_PATIENTS),
            ("derived.trach.invariant_holds", UNIT_BOOL),
            ("derived.trach.invariant_holds_identified", UNIT_BOOL),
            ("derived.trach.invariant_note", UNIT_TEXT),
            ("derived.trach.invariant_scope", UNIT_TEXT),
            ("derived.trach.identified_only_note", UNIT_TEXT),
            ("derived.schema.n_columns", UNIT_COLUMNS),
            ("derived.schema.n_documented", UNIT_COLUMNS),
            ("derived.schema.columns_without_labels", UNIT_COLUMNS),
            ("derived.schema.labels_added", UNIT_COLUMNS),
            ("derived.schema.labels_added_keys", UNIT_TEXT),
        ),
    ),
    (
        "timingfield",
        "The pre-flight namespace holds two different kinds of key. Per OUTCOME, every count "
        "is a count of EVENTS on the denominator ladder and every key ending in pct is a "
        "percentage of that outcome's own row 1, on the 0 to 100 scale. Per FIELD, the "
        "minimum, the maximum, the observed value list and the list of values absent from "
        "the window are DAYS, and the counts beside them are counts of patients carrying "
        "that field. A days-to-event field is bounded to 0 to 30 by the handling rule; a "
        "length-of-stay field is NOT bounded and runs to 112 days here, and the field's own "
        "family key says which of the two it is. The window bounds themselves are integer "
        "postoperative days and the bounding threshold is a percentage. The per-field drop "
        "counts are counts of PATIENTS, one row each, but the total beside them is summed "
        "across eleven fields and is therefore a count of VALUES, not of distinct patients.",
        (
            ("timingfield.<outcome>.pod_missing_pct", UNIT_PERCENT),
            ("timingfield.<outcome>.lost_rows_1_to_4_pct", UNIT_PERCENT),
            ("timingfield.<outcome>.lost_los_missing_stillinhosp_pct", UNIT_PERCENT),
            ("timingfield.<outcome>.events", UNIT_EVENTS),
            ("timingfield.<outcome>.pod_present_n", UNIT_EVENTS),
            ("timingfield.<outcome>.pod_missing_n", UNIT_EVENTS),
            ("timingfield.<outcome>.raw_present_n", UNIT_EVENTS),
            ("timingfield.<outcome>.los_present_n", UNIT_EVENTS),
            ("timingfield.<outcome>.both_present_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_to_range_rule_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_rows_1_to_4_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_day_missing_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_los_missing_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_los_missing_stillinhosp_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_los_missing_not_stillinhosp_n", UNIT_EVENTS),
            ("timingfield.<outcome>.lost_stillinhosp_n", UNIT_EVENTS),
            ("timingfield.<outcome>.bound_unconditional_max_n", UNIT_EVENTS),
            ("timingfield.<outcome>.bound_stillinhosp_conditioned_max_n", UNIT_EVENTS),
            ("timingfield.<outcome>.bounding_required", UNIT_BOOL),
            ("timingfield.<outcome>.postdisch_eligible", UNIT_BOOL),
            ("timingfield.<outcome>.lost_los_missing_all_stillinhosp", UNIT_BOOL),
            ("timingfield.<outcome>.event_col", UNIT_TEXT),
            ("timingfield.<outcome>.pod_col", UNIT_TEXT),
            ("timingfield.<outcome>.raw_fields", UNIT_TEXT),
            ("timingfield.<field>.present_pct", UNIT_PERCENT),
            ("timingfield.<field>.min", UNIT_DAYS),
            ("timingfield.<field>.max", UNIT_DAYS),
            ("timingfield.<field>.values", UNIT_DAYS),
            ("timingfield.<field>.values_absent_from_0_30", UNIT_DAYS),
            ("timingfield.<field>.present_n", UNIT_PATIENTS),
            ("timingfield.<field>.missing_n", UNIT_PATIENTS),
            ("timingfield.<field>.noninteger_n", UNIT_PATIENTS),
            ("timingfield.<field>.outside_0_30_n", UNIT_PATIENTS),
            ("timingfield.<field>.dropped_by_range_rule_n", UNIT_PATIENTS),
            ("timingfield.<field>.distinct_n", UNIT_DISTINCT),
            ("timingfield.<field>.range_rule_applied", UNIT_BOOL),
            ("timingfield.<field>.values_truncated", UNIT_BOOL),
            ("timingfield.<field>.values_are_clean_integer_days", UNIT_BOOL),
            ("timingfield.<field>.family", UNIT_TEXT),
            ("timingfield.bounding_threshold_pct", UNIT_PERCENT),
            ("timingfield.range_min", UNIT_POD),
            ("timingfield.range_max", UNIT_POD),
            ("timingfield.total_dropped_by_range_rule_n", UNIT_VALUES),
            ("timingfield.all_fields_clean_integer_days", UNIT_BOOL),
            ("timingfield.loss_all_los_loss_is_stillinhosp", UNIT_BOOL),
            ("timingfield.fields_audited", UNIT_TEXT),
            ("timingfield.los_fields_audited", UNIT_TEXT),
            ("timingfield.outcomes", UNIT_TEXT),
            ("timingfield.rule", UNIT_TEXT),
            ("timingfield.los_rule_note", UNIT_TEXT),
            ("timingfield.loss_decomposition_note", UNIT_TEXT),
            ("timingfield.loss_bounding_handoff", UNIT_TEXT),
            ("timingfield.loss_stillinhosp_column", UNIT_TEXT),
        ),
    ),
    (
        "losfield",
        "Within the length-of-stay namespace the distribution keys, the disagreement deltas "
        "and the listed stays are DAYS of postoperative stay, and they are NOT bounded at "
        "day 30: the maximum here is 112 days, so the rule that bounds an event day at 30 "
        "must never be applied to a length of stay. Everything else is a count of patients, "
        "except the disagreement rate, which is a percentage on the 0 to 100 scale of the "
        "rows where both fields are present, and the three still-in-hospital scoring keys, "
        "which count EVENTS rather than patients. ONE key is a trap: the count of patients "
        "whose recorded postoperative stay was zero is spelled as though it were a number of "
        "days and is a number of PATIENTS.",
        (
            ("losfield.doptodis_median", UNIT_DAYS),
            ("losfield.doptodis_min", UNIT_DAYS),
            ("losfield.doptodis_max", UNIT_DAYS),
            ("losfield.disagree_min_delta", UNIT_DAYS),
            ("losfield.disagree_max_delta", UNIT_DAYS),
            ("losfield.disagree_deltas", UNIT_DAYS),
            ("losfield.stillinhosp_los_le30_values", UNIT_DAYS),
            ("losfield.disagree_pct_of_both_present", UNIT_PERCENT),
            ("losfield.doptodis_zero_days", UNIT_PATIENTS),
            ("losfield.doptodis_gt30_n", UNIT_PATIENTS),
            ("losfield.doptodis_null", UNIT_PATIENTS),
            ("losfield.tothlos_null", UNIT_PATIENTS),
            ("losfield.htooday_null", UNIT_PATIENTS),
            ("losfield.both_present", UNIT_PATIENTS),
            ("losfield.check_present", UNIT_PATIENTS),
            ("losfield.agree_n", UNIT_PATIENTS),
            ("losfield.disagree_n", UNIT_PATIENTS),
            ("losfield.parent_disagree_n", UNIT_PATIENTS),
            ("losfield.parent_both_present", UNIT_PATIENTS),
            ("losfield.los_gt30_n", UNIT_PATIENTS),
            ("losfield.los_gt30_stillinhosp_n", UNIT_PATIENTS),
            ("losfield.los_cat.<level>.n", UNIT_PATIENTS),
            ("losfield.los_cat.missing_n", UNIT_PATIENTS),
            ("losfield.sameday_disch_n", UNIT_PATIENTS),
            ("losfield.sameday_disch_expired_dest_n", UNIT_PATIENTS),
            ("losfield.sameday_disch_mort30_n", UNIT_PATIENTS),
            ("losfield.sameday_disch_not_discharged_home_any_n", UNIT_PATIENTS),
            ("losfield.sameday_disch_stillinhosp_n", UNIT_PATIENTS),
            ("losfield.disch_by_pod1_n", UNIT_PATIENTS),
            ("losfield.disch_by_pod1_expired_dest_n", UNIT_PATIENTS),
            ("losfield.disch_by_pod1_mort30_n", UNIT_PATIENTS),
            ("losfield.disch_by_pod1_not_discharged_home_any_n", UNIT_PATIENTS),
            ("losfield.disch_by_pod1_stillinhosp_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_yes_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_los_missing_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_los_present_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_los_le30_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_los_gt30_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_sameday_disch_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_disch_by_pod1_n", UNIT_PATIENTS),
            ("losfield.stillinhosp_postdisch_observed_n", UNIT_EVENTS),
            ("losfield.stillinhosp_postdisch_scored_pre_n", UNIT_EVENTS),
            ("losfield.stillinhosp_postdisch_scored_post_n", UNIT_EVENTS),
            ("losfield.disagree_all_one_day_longer", UNIT_BOOL),
            ("losfield.reproduces_parent", UNIT_BOOL),
            ("losfield.primary_field", UNIT_TEXT),
            ("losfield.crosscheck_field", UNIT_TEXT),
            ("losfield.stillinhosp_levels_observed", UNIT_TEXT),
            ("losfield.discharge_contamination_note", UNIT_TEXT),
            ("losfield.stillinhosp_note", UNIT_TEXT),
            ("losfield.stillinhosp_bounding_note", UNIT_TEXT),
        ),
    ),
    (
        "inoutfield",
        "The billing-label namespace publishes counts of PATIENTS and one percentage on the "
        "0 to 100 scale, which is the outpatient share of the analytic cohort. The billing "
        "setting is not a discharge measure and none of these keys is a length of stay.",
        (
            ("inoutfield.outpatient_pct", UNIT_PERCENT),
            ("inoutfield.analytic_n", UNIT_PATIENTS),
            ("inoutfield.flag_present_n", UNIT_PATIENTS),
            ("inoutfield.blank_n", UNIT_PATIENTS),
            ("inoutfield.inpatient_n", UNIT_PATIENTS),
            ("inoutfield.outpatient_n", UNIT_PATIENTS),
            ("inoutfield.is_billing_label", UNIT_BOOL),
            ("inoutfield.levels_observed", UNIT_TEXT),
            ("inoutfield.note", UNIT_TEXT),
        ),
    ),
    (
        "dq",
        "The data-quality namespace publishes counts of patients, counts of hematoma events "
        "on the carried-over reconciliations, and percentages on the 0 to 100 scale. TWO "
        "families carry a unit their name does not. The carried-over reoperation day, both "
        "the plan's own figure and the value re-derived on each rung, is an integer "
        "POSTOPERATIVE DAY published under a bare name with no unit in it. The carried-over "
        "share on the same rungs is a percentage, and the opposite-tie variant beside it is "
        "a percentage on the same denominator and not a difference. Nothing in this "
        "namespace is a published result: these are reconciliations against the carried-over "
        "figures, and the timing step and the post-discharge step own the published values.",
        (
            ("dq.carryover.median_pod.plan_figure", UNIT_POD),
            ("dq.carryover.pod01.<version>.<rung>.median", UNIT_POD),
            ("dq.carryover.pod01.<version>.<rung>.pct", UNIT_PERCENT),
            ("dq.carryover.pod01.<version>.<rung>.n", UNIT_HEM_EVENTS),
            ("dq.carryover.pod01.<version>.<rung>.denominator", UNIT_HEM_EVENTS),
            ("dq.carryover.pod01.plan_figure_pct", UNIT_PERCENT),
            ("dq.carryover.postdisch.<version>.<rung>.pct", UNIT_PERCENT),
            ("dq.carryover.postdisch.<version>.<rung>.pct_opposite_tie", UNIT_PERCENT),
            ("dq.carryover.postdisch.<version>.<rung>.post_n", UNIT_HEM_EVENTS),
            ("dq.carryover.postdisch.<version>.<rung>.events", UNIT_HEM_EVENTS),
            ("dq.carryover.postdisch.<version>.<rung>.denominator", UNIT_HEM_EVENTS),
            ("dq.carryover.postdisch.plan_figure_pct", UNIT_PERCENT),
            ("dq.carryover.postdisch.closest_pct", UNIT_PERCENT),
            ("dq.carryover.postdisch.closest_denominator", UNIT_HEM_EVENTS),
            ("dq.carryover.postdisch.reproduces_to_one_decimal", UNIT_BOOL),
            ("dq.carryover.postdisch.closest_rung", UNIT_TEXT),
            ("dq.carryover.postdisch.closest_definition", UNIT_TEXT),
            ("dq.carryover.postdisch.owner", UNIT_TEXT),
            ("dq.carryover.postdisch.note", UNIT_TEXT),
            ("dq.carryover.owner_note", UNIT_TEXT),
            ("dq.expired.n", UNIT_PATIENTS),
            ("dq.expired.<check>_n", UNIT_PATIENTS),
            ("dq.expired.note", UNIT_TEXT),
            ("dq.hematoma_pod.raw_present_n", UNIT_HEM_EVENTS),
            ("dq.hematoma_pod.usable_present_n", UNIT_HEM_EVENTS),
            ("dq.hematoma_pod.dropped_by_range_rule_n", UNIT_HEM_EVENTS),
            ("dq.hematoma_pod.matches_parent_form", UNIT_BOOL),
            ("dq.mort30.n", UNIT_PATIENTS),
            ("dq.mort30.<check>_n", UNIT_PATIENTS),
            ("dq.mort30.dischdest_levels", UNIT_TEXT),
            ("dq.negflag_days.<outcome>.n", UNIT_PATIENTS),
            ("dq.negflag_days.<outcome>.expected", UNIT_BOOL),
            ("dq.negflag_days.<outcome>.event_col", UNIT_TEXT),
            ("dq.negflag_days.<outcome>.reason", UNIT_TEXT),
            ("dq.negflag_days.<outcome>.raw_fields", UNIT_TEXT),
            ("dq.negflag_days.strict.<field>.n", UNIT_PATIENTS),
            ("dq.negflag_days.strict.<field>.gate_positive_n", UNIT_PATIENTS),
            ("dq.negflag_days.strict.<field>.gate_positive_days_missing_n", UNIT_PATIENTS),
            ("dq.negflag_days.strict.<field>.no_readmission_flag_n", UNIT_PATIENTS),
            ("dq.negflag_days.strict.<field>.planned_readmission_n", UNIT_PATIENTS),
            ("dq.negflag_days.strict.<field>.gate", UNIT_TEXT),
            ("dq.negflag_days.strict_fields", UNIT_TEXT),
            ("dq.negflag_days.unexplained_total_n", UNIT_PATIENTS),
            ("dq.second_event.any_pct", UNIT_PERCENT),
            ("dq.second_event.any_n", UNIT_PATIENTS),
            ("dq.second_event.<slot>.flag_n", UNIT_PATIENTS),
            ("dq.second_event.<slot>.days_present_n", UNIT_PATIENTS),
            ("dq.second_event.<slot>.days_missing_given_flag_n", UNIT_PATIENTS),
            ("dq.second_event.<slot>.unplanned_n", UNIT_PATIENTS),
            ("dq.second_event.<slot>.timing_field_exists", UNIT_BOOL),
            ("dq.second_event.note", UNIT_TEXT),
        ),
    ),
    (
        "prespec",
        "The pre-specification namespace is mostly locked text. Its ONE trap is the risk "
        "threshold list, which is held as PROPORTIONS on the 0 to 1 scale, with a matching "
        "list of rendered percentage labels beside it; multiplying a label by 100 or reading "
        "a threshold as a percentage is the error this declaration exists to prevent. The "
        "follow-up length is a number of days, the timing range bounds are integer "
        "postoperative days, the suppression rule is a count of EVENTS, the bootstrap size "
        "is a count of replicates, and the seed is a dimensionless integer that is not a "
        "measurement of anything.",
        (
            ("prespec.risk_thresholds", UNIT_PROPORTION),
            ("prespec.followup_days", UNIT_DAYS),
            ("prespec.timing_range_min", UNIT_POD),
            ("prespec.timing_range_max", UNIT_POD),
            ("prespec.min_stratum_events", UNIT_EVENTS),
            ("prespec.bootstrap_replicates", UNIT_REPLICATES),
            ("prespec.inout_denominator_n", UNIT_PATIENTS),
            ("prespec.seed", UNIT_SEED),
            ("prespec.no_hematoma_risk_model", UNIT_BOOL),
            ("prespec.no_points_score", UNIT_BOOL),
            ("prespec.steroid_excluded", UNIT_BOOL),
            ("prespec.risk_threshold_labels", UNIT_TEXT),
            ("prespec.stratifiers", UNIT_TEXT),
            ("prespec.stratifier_levels.<stratifier>", UNIT_TEXT),
            ("prespec.timed_outcomes", UNIT_TEXT),
            ("prespec.window_components", UNIT_TEXT),
            ("prespec.window_excluded", UNIT_TEXT),
            ("prespec.postdisch_ineligible_outcomes", UNIT_TEXT),
            ("prespec.bothways_covariates", UNIT_TEXT),
            ("prespec.bootstrap_resampling_unit", UNIT_TEXT),
            ("prespec.censoring", UNIT_TEXT),
            ("prespec.composite_timing", UNIT_TEXT),
            ("prespec.estimator", UNIT_TEXT),
            ("prespec.extrapolation", UNIT_TEXT),
            ("prespec.hematoma_definition", UNIT_TEXT),
            ("prespec.inout_denominator", UNIT_TEXT),
            ("prespec.locked_date", UNIT_TEXT),
            ("prespec.los_field_crosscheck", UNIT_TEXT),
            ("prespec.los_field_primary", UNIT_TEXT),
            ("prespec.mortality_primary", UNIT_TEXT),
            ("prespec.mortality_sensitivity", UNIT_TEXT),
            ("prespec.stratum_comparison", UNIT_TEXT),
            ("prespec.suppression_rule", UNIT_TEXT),
            ("prespec.tie_convention", UNIT_TEXT),
            ("prespec.tie_convention_rule", UNIT_TEXT),
            ("prespec.tie_convention_sensitivity", UNIT_TEXT),
            ("prespec.tie_convention_sensitivity_rule", UNIT_TEXT),
            ("prespec.time_zero", UNIT_TEXT),
            ("prespec.timing_out_of_range_rule", UNIT_TEXT),
            ("prespec.tracheostomy_slot", UNIT_TEXT),
            ("prespec.window_composite", UNIT_TEXT),
        ),
    ),
)

# The four keys each declaration publishes about itself. They describe the declaration and
# are not measurements, so the coverage check below skips them rather than declaring a
# family for a family list.
SCALE_META_SUFFIXES = (".scale", ".scale.families", ".scale.units",
                       ".scale.other_numeric_keys")


def _scale_family_regex(pattern: str):
    """Turn a family such as `cohort.<rung>.rate_pct` into a matcher.

    A `<...>` placeholder matches WITHIN one dotted segment only, so a wildcard cannot
    silently swallow a deeper key. Identical to the matcher `07_figures.py` builds from the
    published lists, which is what makes the published families usable by a gate.
    """
    out = []
    for part in pattern.split("."):
        seg = ""
        for token in re.split(r"(<[^>]*>)", part):
            seg += r"[^.]*" if token.startswith("<") and token.endswith(">") \
                else re.escape(token)
        out.append(seg)
    return re.compile(r"\.".join(out) + r"$")


def publish_scale_declarations(results: dict) -> None:
    """Declare the unit of every numeric family this step publishes.

    Binding: `.build_facts.md` section 9, and section 9.6 specifically, which records that
    this step's nine namespaces were the gap. One declaration per namespace, each scoped to
    its own namespace and saying so, because the merged `results.json` resolves the generic
    interval suffixes to at least four incompatible scales and no step's rule is project
    wide.
    """
    for namespace, prose, families in SCALE_DECLARATIONS:
        results[f"{namespace}.scale"] = f"{prose} {SCALE_SCOPE_NOTE}"
        results[f"{namespace}.scale.families"] = [f for f, _ in families]
        results[f"{namespace}.scale.units"] = [u for _, u in families]
        results[f"{namespace}.scale.other_numeric_keys"] = SCALE_CATCH_ALL


def assert_scale_coverage(results: dict) -> int:
    """Every published key resolves to exactly one declared family. Raises otherwise.

    This is what makes `.build_facts.md` 9.2 item 4 satisfiable without a special case: a
    gate can ask this fragment for the unit of ANY key it holds and get one answer. A new
    key added without a family fails here, inside the step that created it, rather than in
    a downstream gate that cannot know what the key means.
    """
    declared = {ns: [(_scale_family_regex(f), f, u) for f, u in fams]
                for ns, _prose, fams in SCALE_DECLARATIONS}
    missing_ns, unmatched, ambiguous = [], [], []
    for key in results:
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
            "[02] scale-declaration coverage gap (.build_facts.md 9.2 item 4):\n       "
            + "\n       ".join(problems)
        )
    return _scale_family_count()


# --- import-time check: the declared namespaces ARE the reserved namespaces ---
_declared_ns = tuple(ns for ns, _p, _f in SCALE_DECLARATIONS)
if sorted(_declared_ns) != sorted(p.rstrip(".") for p in RESERVED_PREFIXES):
    raise SystemExit(
        f"[02] SCALE_DECLARATIONS covers {sorted(_declared_ns)} but this step reserves "
        f"{sorted(p.rstrip('.') for p in RESERVED_PREFIXES)}. Every reserved namespace "
        f"declares its own units (.build_facts.md 9.2 item 2)."
    )
if len(set(_declared_ns)) != len(_declared_ns):
    raise SystemExit(f"[02] SCALE_DECLARATIONS declares a namespace twice: {_declared_ns}")


# ===========================================================================
# small helpers
# ===========================================================================
def _num(series: pd.Series) -> pd.Series:
    """Bundle text -> float. Empty string is SQL NULL and becomes NaN."""
    return pd.to_numeric(series.astype(str).str.strip().replace({"": None}), errors="coerce")


def _text(series: pd.Series) -> pd.Series:
    """Bundle text, whitespace-stripped. Empty string stays empty (SQL NULL)."""
    return series.astype(str).str.strip()


def _i8(values, index) -> pd.Series:
    """Nullable Int8 on `index`, from a boolean/float array. NaN survives as pd.NA.

    The index is REQUIRED rather than inferred: a Series built from a numpy array carries
    a fresh RangeIndex, and assigning that into a frame aligns by label. Passing the
    frame's own index makes the alignment explicit instead of accidentally correct.
    """
    return pd.Series(np.asarray(values), index=index).astype("Float64").astype("Int8")


def _yesno(series: pd.Series) -> pd.Series:
    """NSQIP Yes/No text -> nullable Int8. Anything that is not Yes or No is missing."""
    v = _text(series).str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="Float64")
    out[v.eq("yes")] = 1.0
    out[v.eq("no")] = 0.0
    return out.astype("Int8")


def _complication(series: pd.Series) -> pd.Series:
    """NSQIP complication text -> nullable Int8, recoded as `!= 'No Complication'`.

    Testing `== 'Yes'` yields an all-zero outcome column (CLAUDE.md 6e.3). Blank is the
    SQL NULL and stays missing.
    """
    v = _text(series)
    out = pd.Series(np.nan, index=series.index, dtype="Float64")
    out[v.eq(NO_COMPLICATION)] = 0.0
    out[v.ne(NO_COMPLICATION) & v.ne("")] = 1.0
    return out.astype("Int8")


def _flag_col(series: pd.Series, predicate) -> pd.Series:
    """Apply a cpt_acdf scalar predicate to a column, cached over its unique values."""
    values = _text(series)
    cache = {v: bool(predicate(v)) for v in values.unique()}
    return values.map(cache)


def _row_any(frame: pd.DataFrame, columns, predicate) -> np.ndarray:
    """True where any of `columns` in the row satisfies `predicate`. NULL-safe."""
    hit = np.zeros(len(frame), dtype=bool)
    for col in columns:
        hit |= _flag_col(frame[col], predicate).to_numpy(dtype=bool)
    return hit


def _positive(series: pd.Series) -> pd.Series:
    """A nullable binary column as a plain boolean mask. Missing counts as NOT positive."""
    return series.astype("boolean").fillna(False).astype(bool)


def _still_in_hospital(a: pd.DataFrame) -> pd.Series:
    """`STILLINHOSP` as a boolean mask, case-insensitively (CLAUDE.md 9.13).

    The column means "still in hospital beyond 30 days after the operation", so a positive
    value says the patient was NEVER discharged inside the follow-up window. That is the
    fact the row-1-to-4 loss decomposition and `04`'s bounding sensitivity turn on, and it
    is read through one helper so every count in this file uses the same definition.
    """
    return _text(a["STILLINHOSP"]).str.casefold().eq("yes")


def usable_days(series: pd.Series) -> pd.Series:
    """Apply THE timing-field handling rule to a raw days-to-event column.

    Returns float64 with the unusable values set to NaN. See TIMING_RANGE_RULE above: the
    rule is applied exactly once, here, and every downstream step inherits it through the
    frozen `POD_*` columns. Nothing is clipped and nothing is silently dropped; the counts
    are published per field under `timingfield.*`.
    """
    v = pd.to_numeric(series, errors="coerce").astype(float)
    arr = v.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(arr) & (arr >= TIMING_DAY_MIN) & (arr <= TIMING_DAY_MAX)
              & (np.mod(arr, 1.0) == 0.0))
    return v.where(pd.Series(ok, index=series.index))


def _fnum(value) -> float:
    """A numpy or pandas scalar as a plain float, with pd.NA folded to NaN."""
    if value is None or value is pd.NA:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pct(numer, denom) -> float:
    return round(100.0 * numer / denom, 4) if denom else float("nan")


# --- code-set rendering (CLAUDE.md hard rule 5) ----------------------------
# `cohort_schema.md` and `derived/notes_02.md` describe what each derived column matches,
# and those descriptions used to spell the CPT and ICD codes out as literals. A literal in
# a description string is exactly as capable of drifting from `cpt_acdf.py` as a literal in
# a predicate, and a reviewer reading the schema has no way to tell that it has. The three
# helpers below render the AUTHORITATIVE sets, so a code can only ever be typed once, in
# `cpt_acdf.py`. Nothing here is a predicate; the predicates already import from `cpt_acdf`.
def cpt_list(codes, sep: str = "/") -> str:
    """A cpt_acdf CPT set as sorted text, joined by `sep`."""
    return sep.join(sorted(codes))


def icd_display(code: str) -> str:
    """A normalized ICD code back into its printed form, decimal point after 3 characters.

    `cpt_acdf` stores every ICD code normalized (no period, uppercase) because that is the
    space all comparisons happen in. The printed form puts the point after the 3-character
    category, which is the rule for both vocabularies: 'AAABB' renders as 'AAA.BB'. Codes of
    three characters or fewer are returned unchanged. This is a formatting rule applied to
    the authoritative sets, not a place where a code is written down.
    """
    text = str(code).strip().upper()
    return text if len(text) <= 3 else f"{text[:3]}.{text[3:]}"


def icd_list(codes, sep: str = "/") -> str:
    """A cpt_acdf ICD set as sorted printed codes, joined by `sep`."""
    return sep.join(icd_display(c) for c in sorted(codes))


# The three pre-specified hematoma code sets differ from one another, and the schema has to
# SAY how without typing a code. These three helpers derive each difference from
# `cpt_acdf.HEMATOMA_ICD10_VARIANTS`, so a code is only ever written in that module.
def _variant_terms(variant: str) -> frozenset:
    spec = CPT.HEMATOMA_ICD10_VARIANTS[variant]
    return frozenset(spec["exact"]) | frozenset(spec["prefixes"])


def _variant_extra(variant: str) -> frozenset:
    """Codes the variant matched that the PRIMARY definition does not."""
    return _variant_terms(variant) - _variant_terms(CPT.HEMATOMA_DEFINITION)


def _variant_missing(variant: str) -> frozenset:
    """Codes the PRIMARY definition matches that the variant does not."""
    return _variant_terms(CPT.HEMATOMA_DEFINITION) - _variant_terms(variant)


def _variant_guard_gap(variant: str) -> frozenset:
    """Seroma codes the PRIMARY guard holds that the variant's guard does not."""
    return (frozenset(CPT.HEMATOMA_ICD10_VARIANTS[CPT.HEMATOMA_DEFINITION]["seroma"])
            - frozenset(CPT.HEMATOMA_ICD10_VARIANTS[variant]["seroma"]))


# ===========================================================================
# 1. read the bundle
# ===========================================================================
def read_bundle(path: pathlib.Path = EXPORT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"[02] export bundle not found at {path}\n"
            f"     Run: python3 01_export_cohort.py"
        )
    # keep_default_na=False is LOAD-BEARING: PRSEPIS stores the literal string 'None' for
    # "no preoperative sepsis" and pandas would otherwise read it as NaN, inverting the
    # sepsis filter (CLAUDE.md 6e.6 / 6d.8). ANESTHES = 'None' has the same shape.
    d = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if d.shape[1] != EXPECTED_BUNDLE_COLS:
        raise SystemExit(
            f"[02] bundle has {d.shape[1]} columns, expected {EXPECTED_BUNDLE_COLS} "
            f"({PARENT_BUNDLE_COLS} in the parent plus {list(GATE1_ADDED_COLUMNS)}). "
            f"Re-run 01_export_cohort.py --force."
        )
    if len(d) != CPT.COHORT_LADDER_N["identified"]:
        raise SystemExit(
            f"[02] bundle has {len(d):,} rows, expected "
            f"{CPT.COHORT_LADDER_N['identified']:,}. CLAUDE.md 8.5: a mismatch is a stop "
            f"condition, not a number to overwrite."
        )
    if (_text(d["PRSEPIS"]) == NO_SEPSIS).sum() == 0:
        raise SystemExit(
            "[02] PRSEPIS has zero rows equal to the literal 'None'. The CSV was read "
            "with pandas NA conversion on; keep_default_na=False is mandatory."
        )
    absent = [c for c in GATE1_ADDED_COLUMNS if c not in d.columns]
    if absent:
        raise SystemExit(
            f"[02] the bundle is missing {absent}. This project's variable list is the "
            f"parent's plus exactly that column (.build_facts.md section 4)."
        )
    return d


# ===========================================================================
# 2. derive
# ===========================================================================
def derive(d: pd.DataFrame, gate1_rebuild: bool = False) -> pd.DataFrame:
    """Add every derived column. Names match variable_dict.LABELS keys verbatim.

    `gate1_rebuild=True` is GATE 1's strengthened structural leg (CLAUDE.md 8.3) and is used
    nowhere else. The frame arrives straight from the raw CSV with every column in
    GATE1_ADDED_COLUMNS DELETED, so the one block below that legitimately reads the added
    column -- the billing-label build, which no eligibility predicate touches -- is skipped
    and EVERY OTHER read of it anywhere in this function would raise KeyError. Nothing else
    changes, so the rungs this mode produces are the real rungs from the real code path, not
    a second copy of the eligibility logic that could drift.
    """
    d = d.copy()

    # --- numeric coercion ---------------------------------------------------
    for col in NUMERIC_COLS:
        d[col] = _num(d[col])
    d["AGE"] = d["AGE"].astype("Int16")

    # Year handling (CLAUDE.md 9.2). The warehouse year is delivered in the bundle as
    # PUFYEAR (the curated view COALESCEs the in-file 2012-2014 hole with puf_year).
    # OPERYR is asserted identical and is set from the same source so the FEATURE_PRETTY
    # key `OPERYR` and the era key `PUFYEAR` can never disagree.
    if d["PUFYEAR"].isna().any():
        raise SystemExit("[02] PUFYEAR has NULLs; the curated view was supposed to COALESCE them.")
    disagree_year = int((d["OPERYR"] != d["PUFYEAR"]).sum())
    if disagree_year:
        raise SystemExit(
            f"[02] OPERYR disagrees with PUFYEAR on {disagree_year:,} rows. They agreed on "
            f"all {len(d):,} rows when the contract was verified. Investigate before "
            f"choosing a year axis (CLAUDE.md 9.2)."
        )
    d["PUFYEAR"] = d["PUFYEAR"].astype("Int16")
    d["OPERYR"] = d["PUFYEAR"]

    # --- eligibility --------------------------------------------------------
    # Elective, NULL-safe on BOTH halves. EMERGNCY stops at 2020 and CASETYPE starts at
    # 2021; a predicate on either alone silently drops the other era.
    emerg = _text(d["EMERGNCY"])
    casetype = _text(d["CASETYPE"])
    late = d["PUFYEAR"] >= 2021
    if (casetype[late] == "").any():
        raise SystemExit(
            "[02] CASETYPE is NULL somewhere in 2021-2024. The COALESCE(...,'Elective') "
            "default would silently admit those cases (CLAUDE.md 9.3)."
        )
    d["ELECTIVE"] = _i8((emerg != "Yes") & ((casetype == "") | (casetype == "Elective")), d.index)

    # Procedure exclusions, scanned across OTHERCPT1..10 ONLY, never CONCPT*.
    oc = list(CPT.OTHERCPT_COLUMNS)
    d["EXCL_CORPECTOMY"] = _i8(_row_any(
        d, oc, lambda c: CPT.normalize_cpt(c) in CPT.EXCLUSION_CORPECTOMY_CPTS), d.index)
    d["EXCL_POSTERIOR"] = _i8(_row_any(
        d, oc, lambda c: CPT.normalize_cpt(c) in CPT.EXCLUSION_POSTERIOR_CERVICAL_CPTS), d.index)
    d["EXCL_ARTHROPLASTY"] = _i8(_row_any(
        d, oc, lambda c: CPT.normalize_cpt(c) in CPT.EXCLUSION_ARTHROPLASTY_CPTS), d.index)
    excl_any = _row_any(d, oc, CPT.is_exclusion_cpt)

    asa_raw = _text(d["ASACLAS"])
    d["EXCL_ASA5"] = _i8(asa_raw == ASA_MORIBUND, d.index)
    d["EXCL_ASA_UNASSIGNED"] = _i8(asa_raw == ASA_UNASSIGNED, d.index)
    d["EXCL_PRESEPSIS"] = _i8(_text(d["PRSEPIS"]) != NO_SEPSIS, d.index)
    d["_EXCL_PROCEDURE"] = excl_any

    # --- demographics -------------------------------------------------------
    d["AGE_PER10"] = d["AGE"].astype("Float64") / 10.0
    sex = _text(d["SEX"]).str.lower()
    d["SEX"] = _text(d["SEX"])
    male = pd.Series(np.nan, index=d.index, dtype="Float64")
    male[sex.eq("male")] = 1.0
    male[sex.eq("female")] = 0.0
    d["MALE"] = male.astype("Int8")           # 'non-binary' is neither; left missing

    # NEW: the locked age stratifier (CLAUDE.md 6.5). The cut points and the level strings
    # are IMPORTED from variable_dict, never typed here, so the parquet levels and the
    # display order cannot drift. Blank where AGE is missing.
    age_f = d["AGE"].astype("Float64")
    age_cat = pd.Series("", index=d.index, dtype=object)
    _age_edges = (18, 50, 65, 75)
    if len(_age_edges) != len(VD.AGE_CAT_ORDER):
        raise SystemExit(
            f"[02] the age cut points and variable_dict.AGE_CAT_ORDER "
            f"{list(VD.AGE_CAT_ORDER)} have different lengths."
        )
    for _lvl, _lo, _hi in zip(VD.AGE_CAT_ORDER, _age_edges, _age_edges[1:] + (None,)):
        sel = age_f.ge(_lo) if _hi is None else (age_f.ge(_lo) & age_f.lt(_hi))
        age_cat[sel.fillna(False).astype(bool)] = _lvl
    d["AGE_CAT"] = age_cat
    _age_seen = sorted(v for v in set(age_cat) if v)
    if _age_seen != sorted(VD.AGE_CAT_ORDER):
        raise SystemExit(
            f"[02] AGE_CAT produced levels {_age_seen}, which do not match "
            f"variable_dict.AGE_CAT_ORDER {sorted(VD.AGE_CAT_ORDER)}."
        )

    race = _text(d["RACE_NEW"])
    eth = _text(d["ETHNICITY_HISPANIC"])
    unmapped = sorted({r for r in race.unique() if r and "," not in r and r not in RACE_MAP})
    if unmapped:
        raise SystemExit(
            f"[02] RACE_NEW carries unmapped level(s) {unmapped}. Extend RACE_MAP "
            f"deliberately rather than letting a new level fall into 'Other'."
        )
    race_cat = race.map(RACE_MAP)
    race_cat = race_cat.where(race_cat.notna(), "Other")     # comma-joined multi-race
    race_cat[race == ""] = "Unknown"
    race_cat[eth == "Yes"] = "Hispanic"                      # ethnicity wins over race
    d["RACE_CAT"] = race_cat
    d["ETHNICITY_HISPANIC"] = eth

    bmi = 703.0 * d["WEIGHT"] / (d["HEIGHT"] ** 2)
    bmi = bmi.replace([np.inf, -np.inf], np.nan)
    implausible = bmi.notna() & ~bmi.between(BMI_MIN, BMI_MAX)
    d["_BMI_IMPLAUSIBLE"] = implausible
    d["BMI"] = bmi.where(~implausible)
    d["BMI_PER5"] = d["BMI"] / 5.0

    # --- comorbidity --------------------------------------------------------
    d["ASACLAS"] = asa_raw
    asa_n = pd.to_numeric(asa_raw.str.slice(0, 1), errors="coerce")
    d["ASA_CAT"] = np.where(asa_n.isna(), "", np.where(asa_n >= 3, ">=3", "1-2"))
    d["ASA_GE3"] = _i8(np.where(asa_n.isna(), np.nan, (asa_n >= 3).astype(float)), d.index)

    fn = _text(d["FNSTATUS2"])
    d["FNSTATUS2"] = fn
    dep = pd.Series(np.nan, index=d.index, dtype="Float64")
    dep[fn.eq("Independent")] = 0.0
    dep[fn.isin(["Partially Dependent", "Totally Dependent"])] = 1.0
    d["FNSTATUS_DEPENDENT"] = dep.astype("Int8")      # 'Unknown' is a real level, not "no"

    dm = _text(d["DIABETES"]).str.upper()
    d["DIABETES"] = _text(d["DIABETES"])
    dany = pd.Series(np.nan, index=d.index, dtype="Float64")
    dany[dm.eq("NO")] = 0.0
    dany[dm.isin(["INSULIN", "NON-INSULIN"])] = 1.0
    d["DIABETES_ANY"] = dany.astype("Int8")

    for col in BINARY_INPLACE:
        d[col] = _yesno(d[col])

    # --- labs and coagulopathy burden --------------------------------------
    d["INR_GT12"] = _i8(np.where(d["PRINR"].isna(), np.nan,
                                 (d["PRINR"] > INR_THRESHOLD).astype(float)), d.index)
    d["PLT_LT150"] = _i8(np.where(d["PRPLATE"].isna(), np.nan,
                                  (d["PRPLATE"] < PLT_THRESHOLD).astype(float)), d.index)
    d["PTT_GT35"] = _i8(np.where(d["PRPTT"].isna(), np.nan,
                                 (d["PRPTT"] > PTT_THRESHOLD).astype(float)), d.index)

    anemia = pd.Series(np.nan, index=d.index, dtype="Float64")
    is_m, is_f = sex.eq("male"), sex.eq("female")
    hct = d["PRHCT"]
    anemia[is_m & hct.notna()] = (hct[is_m & hct.notna()] < ANEMIA_HCT_MALE).astype(float)
    anemia[is_f & hct.notna()] = (hct[is_f & hct.notna()] < ANEMIA_HCT_FEMALE).astype(float)
    d["ANEMIA"] = anemia.astype("Int8")

    d["INR_MEASURED"] = _i8(d["PRINR"].notna(), d.index)
    d["LAB_COMPLETE"] = _i8(d["PRINR"].notna() & d["PRPLATE"].notna() & d["PRPTT"].notna(),
                            d.index)

    # Burden is defined ONLY where the full panel is measured. Anything else would score a
    # missing lab as normal and dilute the exposure. BLEEDDIS is 0% NULL, so LAB_COMPLETE
    # is the binding condition. Components are taken from variable_dict.BURDEN_COMPONENTS.
    comp = list(VD.BURDEN_COMPONENTS)
    burden = sum(d[c].astype("Float64") for c in comp)
    burden = burden.where(d["LAB_COMPLETE"].astype("boolean").fillna(False))
    d["COAG_BURDEN_N"] = burden.astype("Int8")
    d["COAG_BURDEN_PER1"] = burden.astype("Float64")      # same numbers, ordinal model term
    cat = pd.Series("", index=d.index, dtype=object)
    cat[burden.eq(0)] = "0"
    cat[burden.eq(1)] = "1"
    cat[burden.ge(2)] = ">=2"
    d["COAG_BURDEN_CAT"] = cat
    d["COAG_BURDEN_GE2"] = _i8(np.where(burden.isna(), np.nan,
                                        (burden >= 2).astype(float)), d.index)

    # --- procedure ----------------------------------------------------------
    levels = [CPT.count_levels(list(t)) for t in d[oc].itertuples(index=False, name=None)]
    d["LEVELS_N"] = pd.Series(levels, index=d.index, dtype="Int16")
    d["LEVELS_PER_LEVEL"] = d["LEVELS_N"].astype("Float64")
    # CLAUDE.md 6.6. BOTH per-level terms are materialized because the companion project
    # carries both and dropping one would fork the two cohorts. THIS PROJECT FITS NEITHER:
    # it fits no hematoma risk model at all. Capping is a value transform on an existing
    # column, so it cannot move the ladder or any denominator.
    d["LEVELS_PER_LEVEL_CAP3"] = d["LEVELS_N"].clip(upper=LEVELS_CAP).astype("Float64")
    d["LEVELS_CAT"] = [CPT.level_stratum(n) for n in levels]
    d["LEVELS_GE3"] = _i8((d["LEVELS_N"].astype("Float64") >= 3).astype(float), d.index)
    d["ANT_INSTRUM"] = _i8(_row_any(d, oc, CPT.is_anterior_instrumentation_cpt), d.index)

    spec = _text(d["SURGSPEC"])
    d["SURGSPEC"] = spec
    d["SPEC_CAT"] = spec.map(SPEC_MAP).fillna("Other")
    d.loc[spec == "", "SPEC_CAT"] = ""

    # --- primary outcome ----------------------------------------------------
    # Slot 1 only; UNGATED union of all four ICD columns in every year (CLAUDE.md 6e.4).
    # The ICD-10 code set is `cpt_acdf.HEMATOMA_DEFINITION`, V2 (cpt_acdf section 5a). The
    # two other variants are materialized alongside it so `03` can publish them as
    # pre-specified sensitivity rows without re-deriving a cohort.
    hem_cpt = _flag_col(d[CPT.HEMATOMA_CPT_COLUMN], CPT.is_hematoma_cpt)
    hem_i9 = _flag_col(d[CPT.HEMATOMA_ICD9_COLUMN], CPT.is_hematoma_icd9)
    hem_i10 = _flag_col(d[CPT.HEMATOMA_ICD10_COLUMN], CPT.is_hematoma_icd10)
    d["HEMATOMA_CPT_ONLY"] = _i8(hem_cpt.to_numpy(dtype=bool), d.index)
    d["HEMATOMA_ICD_ONLY"] = _i8((hem_i9 | hem_i10).to_numpy(dtype=bool), d.index)
    hematoma = (hem_cpt | hem_i9 | hem_i10).to_numpy(dtype=bool)
    d["HEMATOMA"] = _i8(hematoma, d.index)
    # Cross-check the vectorized union against the scalar API on every row.
    scalar = np.fromiter(
        (CPT.is_hematoma_any(*t) for t in d[[CPT.HEMATOMA_CPT_COLUMN,
                                             CPT.HEMATOMA_ICD9_COLUMN,
                                             CPT.HEMATOMA_ICD10_COLUMN]]
         .itertuples(index=False, name=None)),
        dtype=bool, count=len(d))
    if not np.array_equal(scalar, hematoma):
        raise SystemExit("[02] vectorized hematoma union disagrees with cpt_acdf.is_hematoma_any")
    for _variant in CPT.HEMATOMA_DEFINITION_ORDER:
        if _variant == CPT.HEMATOMA_DEFINITION:
            continue                                  # that one IS `HEMATOMA`
        v10 = _flag_col(d[CPT.HEMATOMA_ICD10_COLUMN],
                        lambda c, _v=_variant: CPT.is_hematoma_icd10_variant(c, _v))
        d[f"HEMATOMA_{_variant}"] = _i8((hem_cpt | hem_i9 | v10).to_numpy(dtype=bool), d.index)
    # The primary definition must be the one cpt_acdf names, and the variant columns must
    # differ from it in exactly the direction section 5a states. A silent switch back to
    # the old code set would otherwise be invisible.
    if CPT.HEMATOMA_DEFINITION != "V2":
        raise SystemExit(
            f"[02] cpt_acdf.HEMATOMA_DEFINITION is {CPT.HEMATOMA_DEFINITION!r}; this "
            f"build expects the V2 primary outcome (CLAUDE.md 6d.9)."
        )
    if not (int(d["HEMATOMA_V1"].sum()) < int(d["HEMATOMA_V0"].sum())
            < int(d["HEMATOMA"].sum())):
        raise SystemExit(
            "[02] the hematoma code-set variants are not ordered V1 < V0 < V2 on the "
            "identified cohort. CLAUDE.md 7.4: that is a stop condition."
        )
    # The roster points the hematoma outcome at THIS column; `POD_HEMATOMA` must not be
    # created (notes_variable_dict.md 7.4). The timing-field handling rule is applied here
    # exactly as it is to every other POD_* column; the raw form is kept internally so the
    # difference between the two can be measured and published rather than assumed to be
    # zero.
    _hem_mask = pd.Series(hematoma, index=d.index)
    d["HEMATOMA_POD"] = usable_days(d["RETORPODAYS"]).where(_hem_mask)
    d["_HEMATOMA_POD_RAW"] = d["RETORPODAYS"].where(_hem_mask)

    # --- secondary outcomes -------------------------------------------------
    d["REOP_ANY"] = _yesno(d["REOPERATION1"])
    if not d["REOP_ANY"].equals(_yesno(d["RETURNOR"])):
        raise SystemExit(
            "[02] RETURNOR is no longer identical to REOPERATION1. The "
            "ported contract says they are the same column in this cohort."
        )
    d["REINTUBATION"] = _complication(d["REINTUB"])
    d["VENT48"] = _complication(d["FAILWEAN"])
    d["PNEUMONIA"] = _complication(d["OUPNEUMO"])
    d["REOP_TRACH"] = _i8(_flag_col(d[CPT.HEMATOMA_CPT_COLUMN],
                                    CPT.is_tracheostomy_cpt).to_numpy(dtype=bool), d.index)
    # NEW (CLAUDE.md 6.2): the wider slots-1-and-2 tracheostomy marker. ASCERTAINMENT
    # SUPPLEMENT ONLY. It is deliberately NOT in AIRWAY_COMP or WINDOW_COMP, so this
    # project's composite stays the parent's frozen slot-1 implementation and the two papers
    # cannot disagree about it. The invariant that the composite is unchanged either way is
    # RE-ASSERTED on every run (trach_invariant below).
    trach_slot2 = _flag_col(d[REOP_SLOT2_CPT_COLUMN],
                            CPT.is_tracheostomy_cpt).to_numpy(dtype=bool)
    d["_TRACH_SLOT2"] = trach_slot2
    d["REOP_TRACH_SLOT12"] = _i8(
        _positive(d["REOP_TRACH"]).to_numpy() | trach_slot2, d.index)

    # CLAUDE.md 6.2 (parent 6b): reintubation OR ventilator > 48 h OR reoperative
    # tracheostomy. Pneumonia is EXCLUDED: parenchymal infection, not an airway-patency
    # event.
    airway = (d["REINTUBATION"].astype("Float64").fillna(0)
              + d["VENT48"].astype("Float64").fillna(0)
              + d["REOP_TRACH"].astype("Float64").fillna(0)) > 0
    d["AIRWAY_COMP"] = _i8(airway.astype(float), d.index)

    # UNPLANNEDREADMISSION1 is populated ONLY when READMISSION1 = 'Yes'; NULL is "no
    # readmission", not "not unplanned" (the ported contract).
    d["READMIT_UNPLANNED"] = _i8((_text(d["UNPLANNEDREADMISSION1"]) == "Yes").astype(float),
                                 d.index)

    # There is NO `DEATH` column (CLAUDE.md 9.4), so mortality has to be derived from
    # DOPERTOD and YRDEATH. CLAUDE.md 9.4 fixes which one is the 30-day flag:
    #
    #   MORT30            DOPERTOD present. DOPERTOD is "days from operation to death" and
    #                     its values are ASSERTED below to lie in [0, 30], so this column
    #                     really is a 30-day flag.
    #   MORT_ANY_RECORDED DOPERTOD present OR YRDEATH present. YRDEATH is a YEAR and is NOT
    #                     window-bounded; among the cases it alone flags, one died on
    #                     postoperative day 33 during the index stay and one was discharged
    #                     ALIVE on day 41. A variable the manuscript calls 30-day mortality
    #                     may not contain deaths provably outside 30 days, so this is a
    #                     SENSITIVITY column and is never labelled 30-day mortality.
    #
    # An earlier revision of this file asserted that both fields are populated only for
    # deaths inside the window. That is true of DOPERTOD and FALSE of YRDEATH; the check
    # below is what the claim should always have been.
    dopertod = d["DOPERTOD"]
    if dopertod.notna().any():
        lo, hi = float(dopertod.min()), float(dopertod.max())
        if lo < 0 or hi > 30:
            raise SystemExit(
                f"[02] DOPERTOD ranges [{lo:g}, {hi:g}], outside the 30-day window that "
                f"MORT30 asserts. CLAUDE.md 7.4: that is a stop condition, not a number "
                f"to clip."
            )
    d["MORT30"] = _i8(dopertod.notna().astype(float), d.index)
    d["MORT_ANY_RECORDED"] = _i8((dopertod.notna() | d["YRDEATH"].notna()).astype(float),
                                 d.index)
    # Cases the union adds over the 30-day flag, and the subset of those provably outside
    # the window. LOS_POSTOP is not assigned yet, so DOPTODIS is read directly.
    d["_YRDEATH_ONLY"] = dopertod.isna() & d["YRDEATH"].notna()
    d["_YRDEATH_ONLY_OUT_OF_WINDOW"] = d["_YRDEATH_ONLY"] & (d["DOPTODIS"] > 30)

    # NEW: the safe-discharge composite (CLAUDE.md 6.2). Exactly HEMATOMA OR AIRWAY_COMP.
    # It gets POD_WINDOW and nothing else: it is not one of the eight ladder outcomes, so it
    # has no POSTDISCH_ or TIMED_BOTH_ column (notes_variable_dict.md 7.4).
    d["WINDOW_COMP"] = _i8(
        ((d["HEMATOMA"].astype("Float64").fillna(0)
          + d["AIRWAY_COMP"].astype("Float64").fillna(0)) > 0).astype(float), d.index)

    # --- length of stay and discharge --------------------------------------
    d["LOS_POSTOP"] = d["DOPTODIS"]           # true postoperative LOS; never INOUT
    # NEW: the independent cross-check (CLAUDE.md 8.7). NEVER the analysis variable.
    d["LOS_POSTOP_CHECK"] = d["TOTHLOS"] - d["HTOODAY"]
    _both_los = d["LOS_POSTOP"].notna() & d["LOS_POSTOP_CHECK"].notna()
    d["LOS_POSTOP_AGREE"] = _i8(
        np.where(_both_los, (d["LOS_POSTOP"] == d["LOS_POSTOP_CHECK"]).astype(float), np.nan),
        d.index)
    _los = d["LOS_POSTOP"]
    los_cat = pd.Series("", index=d.index, dtype=object)
    los_cat[_los.eq(0)] = "0"
    los_cat[_los.eq(1)] = "1"
    los_cat[_los.ge(2)] = ">=2"
    d["LOS_CAT"] = los_cat
    _los_seen = sorted(v for v in set(los_cat) if v)
    if _los_seen != sorted(VD.LOS_CAT_ORDER):
        raise SystemExit(
            f"[02] LOS_CAT produced levels {_los_seen}, which do not match "
            f"variable_dict.LOS_CAT_ORDER {sorted(VD.LOS_CAT_ORDER)}."
        )
    d["SAMEDAY_DISCH"] = _i8(
        np.where(_los.notna(), _los.eq(0).astype(float), np.nan), d.index)
    d["DISCH_BY_POD1"] = _i8(
        np.where(_los.notna(), _los.le(1).astype(float), np.nan), d.index)

    dest = _text(d["DISCHDEST"])
    d["DISCHDEST"] = dest
    # The level map is imported, not re-declared (notes_variable_dict.md 7.3). A blank
    # returns DEST_CAT_MISSING, the empty string, which is exactly what the parent's
    # `.fillna("")` produced; any unmapped non-blank level RAISES.
    try:
        dest_cache = {v: VD.dischdest_category(v) for v in dest.unique()}
    except ValueError as exc:
        raise SystemExit(
            f"[02] {exc}\n     The vocabulary swapped once already at 2021; map a new "
            f"level deliberately in variable_dict.py (CLAUDE.md 9.5)."
        ) from None
    d["DEST_CAT"] = dest.map(dest_cache)
    if VD.DEST_CAT_MISSING != "":
        raise SystemExit(
            "[02] variable_dict.DEST_CAT_MISSING is no longer the empty string. The ported "
            "builder writes that marker into DEST_CAT and NONHOME_DISCH treats it as "
            "missing; changing it forks the two cohorts."
        )
    nonhome = pd.Series(np.nan, index=d.index, dtype="Float64")
    nonhome[d["DEST_CAT"].eq("Home")] = 0.0
    nonhome[d["DEST_CAT"].eq("Facility")] = 1.0
    d["NONHOME_DISCH"] = nonhome.astype("Int8")   # 'Other' (AMA/Expired/Unknown) = missing

    # NEW: `INOUT`, the BILLING label, banned in the parent and required here because the
    # measurement critique is one of this paper's two headline findings (PLAN.md 6.4). Two
    # levels, explicit map imported from variable_dict, unmapped level RAISES, NULL-safe for
    # the single SQL NULL row that exists database-wide. It is NEVER a discharge measure:
    # true same-day discharge is SAMEDAY_DISCH, derived from DOPTODIS.
    # This is the ONLY block in this function that reads the added column, which is what
    # makes Gate 1's rebuild leg a proof rather than an assertion: in that mode the column
    # is absent, this block is skipped, and any other reader raises KeyError.
    if not gate1_rebuild:
        inout = _text(d["INOUT"])
        d["INOUT"] = inout
        try:
            inout_cache = {v: VD.inout_flag(v) for v in inout.unique()}
        except ValueError as exc:
            raise SystemExit(f"[02] {exc}") from None
        d["INOUT_OUTPT"] = _i8(
            pd.to_numeric(inout.map(inout_cache), errors="coerce").to_numpy(dtype=float),
            d.index)

    # NEW: the two coding-era sensitivity strata (PLAN.md 6.6 and 7.3).
    _year = d["PUFYEAR"].astype("Float64")
    d["ERA_ICD10"] = _i8(np.where(_year.notna(), _year.ge(2016).astype(float), np.nan), d.index)
    d["ERA_EMERGNCY"] = _i8(np.where(_year.notna(), _year.le(2020).astype(float), np.nan),
                            d.index)

    # PORTED, never analysed here: CLAUDE.md 7.1 gives the dysphagia analysis to the parent
    # in every form. The columns exist so the two cohorts cannot fork.
    d["DYSPHAGIA_READMIT"] = _i8(_flag_col(
        d[CPT.DYSPHAGIA_READMISSION_ICD10_COLUMN],
        CPT.is_dysphagia_icd10).to_numpy(dtype=bool), d.index)
    d["DYSPHAGIA_REOP"] = _i8(_flag_col(
        d[CPT.DYSPHAGIA_REOPERATION_ICD10_COLUMN],
        CPT.is_dysphagia_icd10).to_numpy(dtype=bool), d.index)
    d["DYSPHAGIA_ANY"] = _i8(((d["DYSPHAGIA_READMIT"].astype("Float64").fillna(0)
                               + d["DYSPHAGIA_REOP"].astype("Float64").fillna(0)) > 0
                              ).astype(float), d.index)

    # --- THE TIMING COLUMNS (new in this project) --------------------------
    # Rule (.build_facts.md 5.2): a POD_* value is taken from the raw days-to-event field
    # ONLY where the corresponding event flag is positive. A days-to-event value present on
    # a row whose event flag is negative is a data-quality finding: it is counted and
    # reported under `dq.negflag_days.*`, and it never reaches a POD_* column.
    #
    # The out-of-range handling rule (TIMING_RANGE_RULE) is applied here, once, through
    # `usable_days`. No downstream step clips or drops.
    d["POD_REOP_ANY"] = usable_days(d["RETORPODAYS"]).where(_positive(d["REOP_ANY"]))
    d["POD_REINTUB"] = usable_days(d["DREINTUB"]).where(_positive(d["REINTUBATION"]))
    d["POD_VENT48"] = usable_days(d["DFAILWEAN"]).where(_positive(d["VENT48"]))
    d["POD_PNEUMONIA"] = usable_days(d["DOUPNEUMO"]).where(_positive(d["PNEUMONIA"]))
    d["POD_READMIT"] = usable_days(d["READMPODAYS1"]).where(_positive(d["READMIT_UNPLANNED"]))
    d["POD_DEATH"] = usable_days(d["DOPERTOD"]).where(_positive(d["MORT30"]))
    d["POD_TRACH"] = usable_days(d["RETORPODAYS"]).where(_positive(d["REOP_TRACH"]))
    # The wider marker's day: the earliest reoperation day across both slots. A tracheostomy
    # after a hematoma evacuation is by construction the patient's SECOND reoperation, so
    # slot 1 cannot see it and slot 2 carries its own day field.
    _t12 = pd.concat([
        usable_days(d["RETORPODAYS"]).where(_positive(d["REOP_TRACH"])),
        usable_days(d[REOP_SLOT2_PODAYS_COLUMN]).where(pd.Series(trach_slot2, index=d.index)),
    ], axis=1)
    d["POD_TRACH12"] = _t12.min(axis=1)

    # CLAUDE.md 6.2: composite timing is the EARLIEST observed day among the components that
    # actually fired. A composite that fired but carries no observed component day stays
    # MISSING; it is never imputed and never given a boundary value.
    d["POD_AIRWAY"] = pd.concat(
        [d["POD_REINTUB"], d["POD_VENT48"], d["POD_TRACH"]], axis=1).min(axis=1)
    d["POD_WINDOW"] = pd.concat(
        [d["HEMATOMA_POD"], d["POD_AIRWAY"]], axis=1).min(axis=1)

    # The two generated families, one per timed outcome. The suffixes come from the ROSTER,
    # not from the event-column name (notes_variable_dict.md 7.4), so they are built through
    # the accessors and never spelled out here.
    _los_present = d["LOS_POSTOP"].notna().to_numpy()
    for _o in VD.TIMED_OUTCOMES:
        if _o.pod_col not in d.columns:
            raise SystemExit(
                f"[02] roster outcome {_o.key!r} names missing column {_o.pod_col}")
        _pod = d[_o.pod_col]
        _both = _pod.notna().to_numpy() & _los_present
        _fired = _positive(d[_o.event_col]).to_numpy()
        with np.errstate(invalid="ignore"):
            _after = (_pod.to_numpy(dtype=float)
                      > d["LOS_POSTOP"].to_numpy(dtype=float))
        # CLAUDE.md 6.3, LOCKED: POSTDISCH = (POD > LOS_POSTOP). An event on the same
        # postoperative day as discharge counts as PRE-discharge, which is the conservative
        # choice for a safety claim. The OPPOSITE convention is NOT materialised here;
        # 04_postdischarge.py computes it inline from these frozen columns, which
        # CLAUDE.md 4.1 grants as the single explicit exception.
        d[VD.postdisch_column(_o.key)] = _i8(
            np.where(_both, _after.astype(float), np.nan), d.index)
        # Row 4 of the PLAN.md 6.2 denominator ladder, defined CONDITIONAL ON THE EVENT:
        # 1 where the event fired and both fields are present, 0 where the event fired and
        # at least one is missing, MISSING where the event did not fire. Summing it gives
        # row 4 directly; counting its zeros gives the events lost between rows 1 and 4.
        d[VD.timed_both_column(_o.key)] = _i8(
            np.where(_fired, _both.astype(float), np.nan), d.index)

    # --- subgroup / denominator flags --------------------------------------
    # PORTED COMPLETENESS ROSTER, NOT A MODEL (notes_variable_dict.md 7.1). The parent read
    # `VD.PRIMARY_MODEL_TERMS`, which is deleted in this project. `MODEL_COMPLETE_TERMS`
    # carries the same six columns in the same order, so this flag reproduces byte for byte
    # and the two cohorts do not fork. Nothing is fitted on it here.
    model_terms = list(VD.MODEL_COMPLETE_TERMS)
    complete = pd.Series(True, index=d.index)
    for term in model_terms:
        complete &= d[term].notna()
    d["MODEL_COMPLETE"] = _i8(complete.astype(float), d.index)

    return d


# ===========================================================================
# 3. the ladder
# ===========================================================================
def ladder_masks(d: pd.DataFrame, replicate_plan: bool) -> "OrderedDict[str, pd.Series]":
    """The four cohort rungs as boolean masks over the full 124,344-row frame.

    `replicate_plan=True` reproduces the plan's strict `TRY_CAST(AGE AS INT) >= 18`, which
    evaluates the raw PUF token '90+' to NULL and drops those patients. The curated view
    already normalized '90+' to 90, so the bundle contains them; the strict cast is
    reproduced by treating the top-coded value as missing. Exactly the rows the plan drops,
    and the count is asserted against cpt_acdf.COHORT_LADDER_NONAGENARIAN_DELTA.
    """
    age = d["AGE"].astype("Float64")
    if replicate_plan:
        age = age.where(age < TOPCODED_AGE)      # strict cast: '90+' -> NULL

    m1 = (age >= MIN_AGE).fillna(False).astype(bool)
    m2 = m1 & d["ELECTIVE"].astype("boolean").fillna(False).astype(bool)
    m3 = m2 & ~pd.Series(d["_EXCL_PROCEDURE"].to_numpy(dtype=bool), index=d.index)
    # ASA is a KEEP list, not a drop list, so both '5-Moribund' and the undocumented
    # 'None assigned' leave the cohort and neither can be reintroduced by a NULL.
    asa_ok = _text(d["ASACLAS"]).isin(ASA_KEEP)
    sepsis_ok = _text(d["PRSEPIS"]).eq(NO_SEPSIS)
    m4 = m3 & asa_ok & sepsis_ok
    return OrderedDict(zip(LADDER_ORDER, [m1, m2, m3, m4]))


def ladder_counts(d: pd.DataFrame, masks) -> "OrderedDict[str, tuple]":
    hem = d["HEMATOMA"].astype("boolean").fillna(False)
    return OrderedDict(
        (step, (int(m.sum()), int((m & hem).sum()))) for step, m in masks.items()
    )


def ladder_verdict(counts, expected_n, expected_events) -> bool:
    """Does this ladder reproduce the pinned cpt_acdf constants, rung for rung?

    The VERDICT a gate publishes, computed from the same comparison `assert_ladder` makes
    and from nothing else. `assert_ladder` stays the stop condition; this is the key that
    RECORDS the result, so `notes_02.md`'s gate table renders a measured verdict and is
    capable of printing FAIL. A literal `True` there would make the table decorative.
    """
    return all(counts[step][0] == expected_n[step]
               and counts[step][1] == expected_events[step]
               for step in LADDER_ORDER)


def assert_ladder(counts, expected_n, expected_events, label: str) -> None:
    bad = []
    for step in LADDER_ORDER:
        n, ev = counts[step]
        if n != expected_n[step]:
            bad.append(f"{label} n[{step}] = {n:,}, expected {expected_n[step]:,} "
                       f"(delta {n - expected_n[step]:+,})")
        if ev != expected_events[step]:
            bad.append(f"{label} events[{step}] = {ev:,}, expected "
                       f"{expected_events[step]:,} (delta {ev - expected_events[step]:+,})")
    if bad:
        raise SystemExit(
            "[02] LADDER MISMATCH. CLAUDE.md 7.4: a mismatch beyond tolerance is a STOP\n"
            "     CONDITION, not a number to overwrite. Do not tune a rule to make it\n"
            "     match; report it.\n       " + "\n       ".join(bad)
        )


def print_ladder(counts, expected_n, expected_events, title: str) -> None:
    print(f"[02] {title}")
    print(f"[02]   {'step':<12} {'n':>10} {'expected':>10}   {'events':>7} {'expected':>8}"
          f"   {'rate %':>7}")
    print(f"[02]   {'-'*12} {'-'*10} {'-'*10}   {'-'*7} {'-'*8}   {'-'*7}")
    for step in LADDER_ORDER:
        n, ev = counts[step]
        print(f"[02]   {step:<12} {n:>10,} {expected_n[step]:>10,}   {ev:>7,} "
              f"{expected_events[step]:>8,}   {pct(ev, n):>7.3f}")


# ===========================================================================
# 4. GATE 1 -- the extended export must not move the cohort (CLAUDE.md 8.3)
# ===========================================================================
def gate_one(d: pd.DataFrame, counts_b, counts_a, n_columns: int) -> dict:
    """Adding `INOUT` to the variable list must not change a single row count.

    Four independent legs, all of which must hold:

      1. WIDTH. The bundle is the parent's width plus exactly the added column(s).
      2. STRUCTURAL, post-derive. Both ladders rebuild identically on the ALREADY-DERIVED
         frame with every added column dropped. This leg proves only that `ladder_masks` and
         `ladder_counts` do not read the added column, because `ELECTIVE` and the `EXCL_*`
         masks are materialised by `derive` before it runs.
      3. STRUCTURAL, full rebuild from the RAW CSV. The bundle is re-read from disk, the
         added column is DELETED before anything looks at it, and both ladders are rebuilt
         through the real `derive` -> `ladder_masks` -> `ladder_counts` path. `derive` reads
         the added column in exactly one block, the billing-label build, which is skipped in
         this mode; any OTHER read of it, at any depth, raises KeyError instead of quietly
         returning a different mask. This is the leg that supports the strong claim, and it
         is the reason legs 2 and 3 are reported separately rather than as one verdict.
      4. NUMERIC. Both ladders land on the pinned cpt_acdf constants, which were established
         on the parent's 142-column bundle. Reproducing them on the 143-column bundle is
         the gate's actual claim.

    T1 additionally proved at the EXPORT layer that deleting `INOUT` from every row of this
    bundle reproduces the parent's bundle byte for byte, with identical SHA-256
    (.build_facts.md 7.1). That proof is recorded here as provenance, not re-run.
    """
    out = {}
    width_ok = n_columns == PARENT_BUNDLE_COLS + len(GATE1_ADDED_COLUMNS)

    stripped = d.drop(columns=[c for c in GATE1_ADDED_COLUMNS if c in d.columns])
    try:
        s_b = ladder_counts(stripped, ladder_masks(stripped, replicate_plan=False))
        s_a = ladder_counts(stripped, ladder_masks(stripped, replicate_plan=True))
        postderive_ok = (s_b == counts_b) and (s_a == counts_a)
        postderive_note = ("both ladders re-counted on the already-derived frame with the "
                           "added column dropped: identical")
    except KeyError as exc:
        postderive_ok = False
        postderive_note = f"the ladder itself reads the added column: {exc}"

    # Leg 3: the full rebuild. The CSV is re-read rather than reusing the in-memory frame so
    # the claim is about the raw bundle and not about a frame this process already touched.
    try:
        raw = read_bundle()
        raw = raw.drop(columns=[c for c in GATE1_ADDED_COLUMNS if c in raw.columns])
        rebuilt = derive(raw, gate1_rebuild=True)
        r_b = ladder_counts(rebuilt, ladder_masks(rebuilt, replicate_plan=False))
        r_a = ladder_counts(rebuilt, ladder_masks(rebuilt, replicate_plan=True))
        rebuild_ok = (r_b == counts_b) and (r_a == counts_a)
        rebuild_note = ("both ladders rebuilt from the raw CSV with the added column deleted "
                        "before any derivation ran: identical, and the added column is "
                        "provably never read outside its own column build")
        for step in LADDER_ORDER:
            out[f"gate.1.rebuild.b.{step}.n"] = int(r_b[step][0])
            out[f"gate.1.rebuild.b.{step}.events"] = int(r_b[step][1])
            out[f"gate.1.rebuild.a.{step}.n"] = int(r_a[step][0])
            out[f"gate.1.rebuild.a.{step}.events"] = int(r_a[step][1])
    except KeyError as exc:
        rebuild_ok = False
        rebuild_note = f"a derivation on the ladder path reads the added column: {exc}"
    out["gate.1.rebuild.added_columns_present"] = False
    out["gate.1.rebuild.from_raw_csv"] = True

    structural_ok = bool(postderive_ok and rebuild_ok)
    numeric_ok = all(counts_b[s][0] == CPT.COHORT_LADDER_N[s]
                     and counts_b[s][1] == CPT.COHORT_LADDER_EVENTS[s]
                     and counts_a[s][0] == CPT.COHORT_LADDER_N_PLAN[s]
                     and counts_a[s][1] == CPT.COHORT_LADDER_EVENTS[s]
                     for s in LADDER_ORDER)
    rows_moved = sum(abs(counts_b[s][0] - CPT.COHORT_LADDER_N[s]) for s in LADDER_ORDER)

    out["gate.1.pass"] = bool(width_ok and structural_ok and numeric_ok)
    out["gate.1.width_pass"] = bool(width_ok)
    out["gate.1.structural_pass"] = bool(structural_ok)
    out["gate.1.structural_postderive_pass"] = bool(postderive_ok)
    out["gate.1.structural_rebuild_pass"] = bool(rebuild_ok)
    out["gate.1.structural_postderive_detail"] = postderive_note
    out["gate.1.structural_rebuild_detail"] = rebuild_note
    out["gate.1.ladder_unchanged"] = bool(numeric_ok)
    out["gate.1.rows_moved"] = int(rows_moved)
    out["gate.1.columns"] = int(n_columns)
    out["gate.1.parent_columns"] = int(PARENT_BUNDLE_COLS)
    out["gate.1.added_columns"] = list(GATE1_ADDED_COLUMNS)
    out["gate.1.detail"] = f"{postderive_note}; {rebuild_note}"
    out["gate.1.note"] = (
        "Adding the billing-setting column to the variable list moved no row count. The "
        "cohort ladder reproduces the pinned constants exactly on the widened bundle; both "
        "ladders rebuild identically from the raw CSV with the added column deleted before "
        "any derivation runs, so no eligibility predicate and no upstream derivation reads "
        "it; and the export layer separately proved this bundle is the parent's bundle plus "
        "exactly one column, row for row."
    )
    out["gate.1.export_proof"] = (
        "T1, .build_facts.md 7.1: 124,344 rows compared, 0 rows differing, column order "
        "identical after removing the added column, SHA-256 identical."
    )
    return out


def print_gate_one(g1: dict) -> None:
    verdict = "PASS" if g1["gate.1.pass"] else "FAIL"
    print("[02] GATE 1 -- the extended export must not move the cohort (CLAUDE.md 8.3)")
    print(f"[02]   bundle width          {g1['gate.1.columns']} = "
          f"{g1['gate.1.parent_columns']} parent + {g1['gate.1.added_columns']}   "
          f"{'OK' if g1['gate.1.width_pass'] else 'FAIL'}")
    print(f"[02]   structural, ladder    {g1['gate.1.structural_postderive_detail']}   "
          f"{'OK' if g1['gate.1.structural_postderive_pass'] else 'FAIL'}")
    print(f"[02]   structural, rebuild   {g1['gate.1.structural_rebuild_detail']}   "
          f"{'OK' if g1['gate.1.structural_rebuild_pass'] else 'FAIL'}")
    print(f"[02]   ladder unchanged      rows moved = {g1['gate.1.rows_moved']}   "
          f"{'OK' if g1['gate.1.ladder_unchanged'] else 'FAIL'}")
    print(f"[02]   GATE 1 {verdict}")
    if not g1["gate.1.pass"]:
        print("[02]   BLOCKER: CLAUDE.md 8.3 makes this a stop condition. The export layer "
              "already proved the bundle is the parent's plus one column, so the cause is "
              "in 02.")


# ===========================================================================
# 5. THE PRE-FLIGHT DATA CHECK (CLAUDE.md 8.6), before any curve is ever drawn
# ===========================================================================
def preflight_timing(a: pd.DataFrame) -> dict:
    """Tabulate every timing field's observed value set, and conditional missingness.

    Three questions, all on the ANALYTIC cohort:

      1. What values does each days-to-event field actually take HERE? The warehouse catalog
         reports 41 distinct values for the reoperation-day field and 48 for the
         readmission-day field across all of `main`, where a clean 0 to 30 integer range
         would give 31. This function measures the value set inside this cohort.
      2. How is anything outside 0 to 30, or anything non-integer, handled? By
         TIMING_RANGE_RULE, applied once through `usable_days`, with the counts published
         here. Nothing is clipped and nothing is silently dropped.
      3. For every timed outcome, how missing is the days-to-event field GIVEN THAT THE
         EVENT FLAG IS POSITIVE? That is the number that matters and it is unmeasured in the
         literature. The database-wide percentages in PLAN.md 2.4 are EVENT RATES, not
         missingness rates, and the manuscript must not confuse the two.
      4. WHY is each event lost between ladder rows 1 and 4? The loss is decomposed three
         ways: a missing days-to-event field, a missing LENGTH OF STAY on a patient flagged
         still in hospital beyond 30 days, and a missing length of stay on a patient NOT so
         flagged. The split is what lets 04 bound honestly (PLAN.md 6.2): a patient who was
         never discharged inside the window cannot have had the event after a discharge that
         did not occur, so the naive "all missing events post-discharge" extreme is not
         merely conservative for those rows, it is structurally impossible. NOTHING is
         reclassified here and no ladder count moves; this is the breakdown, not a change.
    """
    out = {}
    n = len(a)

    # --- 1 and 2: the observed value set of every timing field ------------
    for field in TIMING_FIELDS + LOS_FIELDS:
        key = field.lower()
        raw = pd.to_numeric(a[field], errors="coerce").astype(float)
        present = raw.notna()
        vals = np.sort(raw.dropna().unique())
        arr = raw.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            noninteger = present.to_numpy() & (np.mod(arr, 1.0) != 0.0)
            outside = present.to_numpy() & ((arr < TIMING_DAY_MIN) | (arr > TIMING_DAY_MAX))
        is_los = field in LOS_FIELDS
        out[f"timingfield.{key}.family"] = "length of stay" if is_los else "days to event"
        out[f"timingfield.{key}.present_n"] = int(present.sum())
        out[f"timingfield.{key}.missing_n"] = int(n - int(present.sum()))
        out[f"timingfield.{key}.present_pct"] = pct(int(present.sum()), n)
        out[f"timingfield.{key}.distinct_n"] = int(len(vals))
        out[f"timingfield.{key}.min"] = _fnum(vals.min()) if len(vals) else float("nan")
        out[f"timingfield.{key}.max"] = _fnum(vals.max()) if len(vals) else float("nan")
        out[f"timingfield.{key}.noninteger_n"] = int(noninteger.sum())
        out[f"timingfield.{key}.outside_0_30_n"] = int(outside.sum())
        out[f"timingfield.{key}.range_rule_applied"] = bool(not is_los)
        out[f"timingfield.{key}.dropped_by_range_rule_n"] = (
            0 if is_los else int((noninteger | outside).sum()))
        listed = [int(v) if float(v).is_integer() else round(float(v), 4) for v in vals]
        # A list of scalars, which the fragment contract allows. Only the length-of-stay
        # fields are capped, and only because their right tail is long by design.
        if is_los and len(listed) > 60:
            out[f"timingfield.{key}.values_truncated"] = True
            out[f"timingfield.{key}.values"] = listed[:60]
        else:
            out[f"timingfield.{key}.values_truncated"] = False
            out[f"timingfield.{key}.values"] = listed
        if not is_los:
            expected = list(range(TIMING_DAY_MIN, TIMING_DAY_MAX + 1))
            out[f"timingfield.{key}.values_are_clean_integer_days"] = bool(
                set(listed).issubset(set(expected)))
            out[f"timingfield.{key}.values_absent_from_0_30"] = [
                v for v in expected if v not in set(listed)]

    out["timingfield.rule"] = TIMING_RANGE_RULE
    out["timingfield.range_min"] = TIMING_DAY_MIN
    out["timingfield.range_max"] = TIMING_DAY_MAX
    out["timingfield.fields_audited"] = list(TIMING_FIELDS)
    out["timingfield.los_fields_audited"] = list(LOS_FIELDS)
    out["timingfield.total_dropped_by_range_rule_n"] = int(sum(
        out[f"timingfield.{f.lower()}.dropped_by_range_rule_n"] for f in TIMING_FIELDS))
    out["timingfield.all_fields_clean_integer_days"] = bool(all(
        out[f"timingfield.{f.lower()}.values_are_clean_integer_days"] for f in TIMING_FIELDS))
    out["timingfield.los_rule_note"] = (
        "The 0 to 30 day rule applies to days-to-event fields only. Length of stay is not a "
        "window-bounded quantity: a patient may legitimately stay in hospital for more than "
        "30 days, and truncating a length of stay would corrupt the discharge-timing "
        "analysis this project exists to do."
    )

    # --- 3: conditional missingness, per timed outcome --------------------
    los_present = a["LOS_POSTOP"].notna()
    still = _still_in_hospital(a)
    still_all = []
    for o in VD.TIMED_OUTCOMES:
        k = o.key
        fired = _positive(a[o.event_col])
        events = int(fired.sum())
        pod = a[o.pod_col]
        pod_present = int((fired & pod.notna()).sum())
        if k == "airway":
            # The composite has three possible source fields; "present" means at least one
            # component that FIRED carries an observed day.
            comp_raw = ((_positive(a["REINTUBATION"]) & a["DREINTUB"].notna())
                        | (_positive(a["VENT48"]) & a["DFAILWEAN"].notna())
                        | (_positive(a["REOP_TRACH"]) & a["RETORPODAYS"].notna()))
            raw_present = int((fired & comp_raw).sum())
        else:
            raw = pd.to_numeric(a[o.raw_fields[0]], errors="coerce")
            raw_present = int((fired & raw.notna()).sum())
        both = int((fired & pod.notna() & los_present).sum())
        out[f"timingfield.{k}.events"] = events
        out[f"timingfield.{k}.pod_present_n"] = pod_present
        out[f"timingfield.{k}.pod_missing_n"] = events - pod_present
        out[f"timingfield.{k}.pod_missing_pct"] = pct(events - pod_present, events)
        out[f"timingfield.{k}.raw_present_n"] = raw_present
        out[f"timingfield.{k}.lost_to_range_rule_n"] = raw_present - pod_present
        out[f"timingfield.{k}.los_present_n"] = int((fired & los_present).sum())
        out[f"timingfield.{k}.both_present_n"] = both
        out[f"timingfield.{k}.lost_rows_1_to_4_n"] = events - both
        out[f"timingfield.{k}.lost_rows_1_to_4_pct"] = pct(events - both, events)
        out[f"timingfield.{k}.bounding_required"] = bool(
            events > 0 and (events - both) / events > 0.05)
        out[f"timingfield.{k}.event_col"] = o.event_col
        out[f"timingfield.{k}.pod_col"] = o.pod_col
        out[f"timingfield.{k}.raw_fields"] = list(o.raw_fields)
        out[f"timingfield.{k}.postdisch_eligible"] = bool(o.postdisch_eligible)

        # --- 4: WHY the event was lost between rows 1 and 4 ---------------
        # Three disjoint reasons, and they sum to the loss. No count above changes.
        lost_day = int((fired & pod.isna()).sum())
        lost_los_still = int((fired & pod.notna() & ~los_present & still).sum())
        lost_los_not_still = int((fired & pod.notna() & ~los_present & ~still).sum())
        lost_total = events - both
        if lost_day + lost_los_still + lost_los_not_still != lost_total:
            raise SystemExit(
                f"[02] the row 1 to 4 loss decomposition for {k!r} does not sum: "
                f"{lost_day} + {lost_los_still} + {lost_los_not_still} != {lost_total}. "
                f"The three reasons are supposed to be disjoint and exhaustive."
            )
        lost_los = lost_los_still + lost_los_not_still
        out[f"timingfield.{k}.lost_day_missing_n"] = lost_day
        out[f"timingfield.{k}.lost_los_missing_n"] = lost_los
        out[f"timingfield.{k}.lost_los_missing_stillinhosp_n"] = lost_los_still
        out[f"timingfield.{k}.lost_los_missing_not_stillinhosp_n"] = lost_los_not_still
        out[f"timingfield.{k}.lost_los_missing_stillinhosp_pct"] = pct(
            lost_los_still, lost_los)
        # None, not False, where there is no length-of-stay loss to attribute: a False here
        # would read as "some of it is not a still-in-hospital patient", which is not what a
        # zero denominator says.
        out[f"timingfield.{k}.lost_los_missing_all_stillinhosp"] = (
            bool(lost_los_not_still == 0) if lost_los > 0 else None)
        out[f"timingfield.{k}.lost_stillinhosp_n"] = int(
            (fired & ~(pod.notna() & los_present) & still).sum())
        # The events a bound may legitimately place after discharge. A patient still in
        # hospital beyond day 30 was never discharged inside the window, so a lost event on
        # that row CANNOT be post-discharge; a lost event whose LENGTH OF STAY is unknown on
        # any other row, or whose DAY is unknown, still can be.
        out[f"timingfield.{k}.bound_unconditional_max_n"] = lost_total
        out[f"timingfield.{k}.bound_stillinhosp_conditioned_max_n"] = (
            lost_day + lost_los_not_still)
        if lost_los > 0:
            still_all.append(lost_los_not_still == 0)
    out["timingfield.outcomes"] = list(VD.TIMED_OUTCOME_KEYS)
    out["timingfield.bounding_threshold_pct"] = 5.0
    out["timingfield.loss_stillinhosp_column"] = "STILLINHOSP"
    out["timingfield.loss_all_los_loss_is_stillinhosp"] = bool(
        len(still_all) > 0 and all(still_all))
    out["timingfield.loss_decomposition_note"] = (
        "The loss between rows 1 and 4 of the PLAN.md 6.2 denominator ladder is decomposed "
        "into three disjoint reasons: the days-to-event field is missing; the length of "
        "stay is missing on a patient flagged still in hospital beyond 30 days; and the "
        "length of stay is missing on a patient not so flagged. Nothing is reclassified and "
        "no ladder count moves. The split exists because a patient who was never discharged "
        "inside the follow-up window cannot have had the event after a discharge that did "
        "not occur, so for those rows the 'all missing events post-discharge' extreme is "
        "structurally impossible rather than merely conservative."
    )
    out["timingfield.loss_bounding_handoff"] = (
        "Binding on 04_postdischarge.py: report the STILLINHOSP-conditioned bound as the "
        "primary bounding sensitivity, and show the naive unconditional bound only beside "
        "it, labelled as the unconditional extreme."
    )
    return out


def negative_flag_days(a: pd.DataFrame) -> dict:
    """Days-to-event values present on rows whose event flag is NEGATIVE.

    .build_facts.md 5.2: count it, report it, do not use it. None of these values reaches a
    `POD_*` column, because every `POD_*` column is masked by its own event flag.
    """
    out = {}
    for o in VD.TIMED_OUTCOMES:
        fired = _positive(a[o.event_col])
        any_raw = pd.Series(False, index=a.index)
        for f in o.raw_fields:
            any_raw |= pd.to_numeric(a[f], errors="coerce").notna()
        out[f"dq.negflag_days.{o.key}.n"] = int((~fired & any_raw).sum())
        out[f"dq.negflag_days.{o.key}.event_col"] = o.event_col
        out[f"dq.negflag_days.{o.key}.raw_fields"] = list(o.raw_fields)
        out[f"dq.negflag_days.{o.key}.expected"] = bool(o.key in NEGFLAG_EXPECTED)
        out[f"dq.negflag_days.{o.key}.reason"] = NEGFLAG_EXPECTED.get(
            o.key, "no documented reason: an unexpected value here is a data-quality finding")
    for field, gate in STRICT_GATE.items():
        raw = pd.to_numeric(a[field], errors="coerce")
        fired = _positive(a[gate])
        out[f"dq.negflag_days.strict.{field.lower()}.n"] = int((~fired & raw.notna()).sum())
        out[f"dq.negflag_days.strict.{field.lower()}.gate"] = gate
        out[f"dq.negflag_days.strict.{field.lower()}.gate_positive_n"] = int(fired.sum())
        out[f"dq.negflag_days.strict.{field.lower()}.gate_positive_days_missing_n"] = int(
            (fired & raw.isna()).sum())
    # The readmission day is gated by READMISSION1, not by the UNPLANNED flag, so a day on a
    # "negative" row is usually a PLANNED readmission. Measured rather than asserted.
    r1 = _text(a["READMISSION1"]).eq("Yes")
    ru1 = _positive(a["READMIT_UNPLANNED"])
    rday = pd.to_numeric(a["READMPODAYS1"], errors="coerce")
    out["dq.negflag_days.strict.readmpodays1.planned_readmission_n"] = int(
        (~ru1 & rday.notna() & r1).sum())
    out["dq.negflag_days.strict.readmpodays1.no_readmission_flag_n"] = int(
        (~ru1 & rday.notna() & ~r1).sum())
    out["dq.negflag_days.strict_fields"] = [f.lower() for f in STRICT_GATE]
    out["dq.negflag_days.unexplained_total_n"] = int(sum(
        out[f"dq.negflag_days.{o.key}.n"] for o in VD.TIMED_OUTCOMES
        if o.key not in NEGFLAG_EXPECTED))
    return out


def second_events(a: pd.DataFrame) -> dict:
    """Patients with a SECOND event whose timing is only partially observed.

    CLAUDE.md 9.18 and PLAN.md section 4: NSQIP records days-to-event for occurrence 1 in
    most families, plus a second reoperation and a second through fifth readmission, and
    nothing beyond. All analyses are time-to-first-event and this is the count the Methods
    must state.
    """
    out = {}
    reop2_flag = _text(a["REOPERATION2"]).eq("Yes")
    reop3_flag = _text(a["REOPERATION3"]).eq("Yes")
    reop2_days = pd.to_numeric(a[REOP_SLOT2_PODAYS_COLUMN], errors="coerce")
    out["dq.second_event.reop2.flag_n"] = int(reop2_flag.sum())
    out["dq.second_event.reop2.days_present_n"] = int(reop2_days.notna().sum())
    out["dq.second_event.reop2.days_missing_given_flag_n"] = int(
        (reop2_flag & reop2_days.isna()).sum())
    out["dq.second_event.reop3.flag_n"] = int(reop3_flag.sum())
    out["dq.second_event.reop3.timing_field_exists"] = False

    readm_flags = {}
    for i in (2, 3, 4, 5):
        flag = _text(a[f"READMISSION{i}"]).eq("Yes")
        unpl = _text(a[f"UNPLANNEDREADMISSION{i}"]).eq("Yes")
        days = pd.to_numeric(a[f"READMPODAYS{i}"], errors="coerce")
        readm_flags[i] = flag
        out[f"dq.second_event.readmit{i}.flag_n"] = int(flag.sum())
        out[f"dq.second_event.readmit{i}.unplanned_n"] = int(unpl.sum())
        out[f"dq.second_event.readmit{i}.days_present_n"] = int(days.notna().sum())
    any_second = (reop2_flag | reop3_flag | readm_flags[2] | readm_flags[3]
                  | readm_flags[4] | readm_flags[5])
    out["dq.second_event.any_n"] = int(any_second.sum())
    out["dq.second_event.any_pct"] = pct(int(any_second.sum()), len(a))
    out["dq.second_event.note"] = (
        "Only the first event of each type is timed in the analyses. A second reoperation "
        "and a second through fifth readmission carry their own day fields and are counted "
        "here; a third reoperation carries none. All analyses are time-to-first-event, "
        "pre-specified."
    )
    return out


def carryover_reconciliation(d: pd.DataFrame, masks) -> dict:
    """Which cohort rung and which code set does the carried-over 41.3% come from?

    `PLAN.md` 2.2 carries "41.3% of hematoma reoperations occur after that patient's own
    discharge" and `PLAN.md` 13 lists its denominator as an OPEN ITEM: it was never
    recorded. `CLAUDE.md` 8.4 makes it a Gate 2 number and 8.5 makes a mismatch a stop
    condition rather than something to overwrite in either direction.

    The frozen columns this step already produces answer the question directly, so the
    reconciliation is measured here rather than left for a later step to rediscover. Every
    rung is evaluated under both the primary and the superseded code set, with the two tie
    conventions side by side.

    **`04_postdischarge.py` OWNS the published post-discharge fraction and must re-derive
    it.** These keys live in this step's data-quality namespace precisely so they cannot
    collide with, or be mistaken for, the published figure. Nothing here is adopted: the
    manuscript reports the ANALYTIC-cohort figure (`CLAUDE.md` 4.4) whatever this says.
    """
    out = {}
    los = d["LOS_POSTOP"]
    best = None
    for step in LADDER_ORDER:
        for col, tag in (("HEMATOMA", "v2"), ("HEMATOMA_V0", "v0")):
            ev = masks[step] & _positive(d[col])
            pod = usable_days(d["RETORPODAYS"]).where(ev)
            both = ev & pod.notna() & los.notna()
            post = int((both & (pod > los)).sum())
            alt = int((both & (pod >= los)).sum())
            k = f"dq.carryover.postdisch.{tag}.{step}"
            out[f"{k}.events"] = int(ev.sum())
            out[f"{k}.denominator"] = int(both.sum())
            out[f"{k}.post_n"] = post
            out[f"{k}.pct"] = pct(post, int(both.sum()))
            out[f"{k}.pct_opposite_tie"] = pct(alt, int(both.sum()))
            gap = abs(round(pct(post, int(both.sum())), 1) - 41.3)
            if best is None or gap < best[0]:
                best = (gap, step, tag, pct(post, int(both.sum())), int(both.sum()))
    out["dq.carryover.postdisch.plan_figure_pct"] = 41.3
    out["dq.carryover.postdisch.closest_rung"] = best[1]
    out["dq.carryover.postdisch.closest_definition"] = best[2]
    out["dq.carryover.postdisch.closest_pct"] = best[3]
    out["dq.carryover.postdisch.closest_denominator"] = best[4]
    out["dq.carryover.postdisch.reproduces_to_one_decimal"] = bool(
        round(best[3], 1) == 41.3)
    out["dq.carryover.postdisch.owner"] = "04_postdischarge.py"
    out["dq.carryover.postdisch.note"] = (
        "PLAN.md section 13 lists the denominator behind the carried-over 41.3% as an open "
        "item. Measured here from the frozen columns across every rung and both code sets. "
        "04_postdischarge.py owns the published figure and must re-derive it; the "
        "manuscript reports the analytic-cohort value whatever this reconciliation says, "
        "because the analytic cohort is the study population (CLAUDE.md 4.4). A mismatch is "
        "reported, not overwritten in either direction (CLAUDE.md 8.5).")
    # The companion carried-over figure, the share of hematoma reoperations on day 0 or 1.
    for step in LADDER_ORDER:
        for col, tag in (("HEMATOMA", "v2"), ("HEMATOMA_V0", "v0")):
            ev = masks[step] & _positive(d[col])
            pod = usable_days(d["RETORPODAYS"]).where(ev)
            k = f"dq.carryover.pod01.{tag}.{step}"
            out[f"{k}.denominator"] = int(pod.notna().sum())
            out[f"{k}.n"] = int((pod <= 1).sum())
            out[f"{k}.pct"] = pct(int((pod <= 1).sum()), int(pod.notna().sum()))
            out[f"{k}.median"] = _fnum(pod.median())
    out["dq.carryover.pod01.plan_figure_pct"] = 42.6
    out["dq.carryover.median_pod.plan_figure"] = 2.0
    out["dq.carryover.owner_note"] = (
        "03_timing.py owns the published median postoperative day and the day 0 to 1 share; "
        "04_postdischarge.py owns the published post-discharge fraction. These keys are the "
        "reconciliation of the carried-over figures against their unrecorded denominators, "
        "and they are not the published values.")
    return out


def trach_invariant(d: pd.DataFrame, mask_analytic: pd.Series) -> dict:
    """CLAUDE.md 6.2: AIRWAY_COMP must not move when the tracheostomy arm widens to slots 1+2.

    The parent proved that every slot-2 tracheostomy already enters the composite through
    unplanned reintubation or prolonged ventilation, which is why this project can
    materialise the wider marker for the ascertainment supplement while leaving the
    composite on the frozen slot-1 implementation. That claim is RE-ASSERTED on every run
    rather than inherited, and a BLOCKER is printed if it stops holding.

    SCOPE, stated explicitly because the two populations give different answers. The claim
    is about the ANALYTIC cohort: that is the population every analysis uses, and it is
    where the parent measured it (its "all six slot-2 tracheostomies" is the analytic-cohort
    count). `invariant_holds` is therefore the analytic-cohort claim. The identified-cohort
    figure is measured and published BESIDE it, unadjusted, so a wider difference among
    patients the cohort definition excludes is on the record rather than hidden.
    """
    out = {}
    slot12_comp = ((d["REINTUBATION"].astype("Float64").fillna(0)
                    + d["VENT48"].astype("Float64").fillna(0)
                    + d["REOP_TRACH_SLOT12"].astype("Float64").fillna(0)) > 0)
    frozen = _positive(d["AIRWAY_COMP"])
    differ_all = int((slot12_comp.to_numpy() != frozen.to_numpy()).sum())
    a = d[mask_analytic]
    differ_analytic = int(
        (slot12_comp[mask_analytic].to_numpy() != _positive(a["AIRWAY_COMP"]).to_numpy()).sum())
    slot2_only = pd.Series(d["_TRACH_SLOT2"].to_numpy(dtype=bool), index=d.index) & ~_positive(
        d["REOP_TRACH"])
    out["derived.trach.slot1_identified_n"] = int(_positive(d["REOP_TRACH"]).sum())
    out["derived.trach.slot12_identified_n"] = int(_positive(d["REOP_TRACH_SLOT12"]).sum())
    out["derived.trach.slot1_analytic_n"] = int(a["REOP_TRACH"].sum())
    out["derived.trach.slot12_analytic_n"] = int(a["REOP_TRACH_SLOT12"].sum())
    out["derived.trach.slot2_only_identified_n"] = int(slot2_only.sum())
    out["derived.trach.slot2_only_analytic_n"] = int(slot2_only[mask_analytic].sum())
    out["derived.trach.slot2_only_via_reintubation_n"] = int(
        (slot2_only & _positive(d["REINTUBATION"])).sum())
    out["derived.trach.slot2_only_via_vent48_n"] = int(
        (slot2_only & _positive(d["VENT48"])).sum())
    out["derived.trach.slot2_only_already_in_composite_n"] = int(
        (slot2_only & frozen).sum())
    out["derived.trach.slot2_only_analytic_already_in_composite_n"] = int(
        (slot2_only & frozen & mask_analytic).sum())
    out["derived.trach.invariant_rows_differing_identified"] = differ_all
    out["derived.trach.invariant_rows_differing_analytic"] = differ_analytic
    # The claim is about the ANALYTIC cohort; see the docstring. The identified-cohort
    # measurement is published beside it and is never used to adjudicate the gate.
    out["derived.trach.invariant_scope"] = "analytic cohort"
    out["derived.trach.invariant_holds"] = bool(differ_analytic == 0)
    out["derived.trach.invariant_holds_identified"] = bool(differ_all == 0)
    out["derived.trach.invariant_note"] = (
        "On the analytic cohort the composite airway outcome is identical whether the "
        "tracheostomy arm reads reoperation slot 1 only or slots 1 and 2, because every "
        "slot-2 tracheostomy there already enters the composite through unplanned "
        "reintubation or prolonged ventilation. The composite therefore stays on the "
        "parent's frozen slot-1 implementation and the two papers cannot disagree about it."
    )
    out["derived.trach.identified_only_note"] = (
        "Measured on the wider identified cohort the two readings differ on a small number "
        "of patients, every one of whom the cohort definition excludes before any analysis "
        "sees them. That figure is published unadjusted rather than suppressed, but it does "
        "not bear on the composite, which is only ever computed on the analytic cohort."
    )
    return out


def expired_reconciliation(a: pd.DataFrame) -> dict:
    """CLAUDE.md 8.8: reconcile DISCHDEST = 'Expired' against DOPERTOD and the mortality flag.

    A patient who expired has no discharge day. They are a COMPETING EVENT, not a same-day
    discharge, and a length-of-stay field read naively would fold them into the discharge
    distribution. Every mismatch in either direction is reported.
    """
    out = {}
    expired = a["DISCHDEST"].eq("Expired")
    mort = _positive(a["MORT30"])
    mort_any = _positive(a["MORT_ANY_RECORDED"])
    los = a["LOS_POSTOP"]
    dop = pd.to_numeric(a["DOPERTOD"], errors="coerce")
    out["dq.expired.n"] = int(expired.sum())
    out["dq.expired.mort30_n"] = int((expired & mort).sum())
    out["dq.expired.no_mort30_n"] = int((expired & ~mort).sum())
    out["dq.expired.no_mort_any_recorded_n"] = int((expired & ~mort_any).sum())
    out["dq.expired.los_present_n"] = int((expired & los.notna()).sum())
    out["dq.expired.los_missing_n"] = int((expired & los.isna()).sum())
    out["dq.expired.los_zero_n"] = int((expired & los.eq(0)).sum())
    out["dq.expired.stillinhosp_yes_n"] = int((expired & _still_in_hospital(a)).sum())
    out["dq.expired.death_day_after_los_n"] = int(
        (expired & dop.notna() & los.notna() & (dop > los)).sum())
    out["dq.expired.death_day_equals_los_n"] = int(
        (expired & dop.notna() & los.notna() & (dop == los)).sum())
    out["dq.expired.death_day_before_los_n"] = int(
        (expired & dop.notna() & los.notna() & (dop < los)).sum())
    out["dq.mort30.n"] = int(mort.sum())
    out["dq.mort30.dischdest_expired_n"] = int((mort & expired).sum())
    out["dq.mort30.dischdest_not_expired_n"] = int((mort & ~expired).sum())
    out["dq.mort30.dischdest_home_n"] = int((mort & a["DEST_CAT"].eq("Home")).sum())
    out["dq.mort30.dischdest_facility_n"] = int((mort & a["DEST_CAT"].eq("Facility")).sum())
    out["dq.mort30.dischdest_missing_n"] = int((mort & a["DISCHDEST"].eq("")).sum())
    out["dq.mort30.dischdest_levels"] = sorted(
        v for v in a.loc[mort & a["DISCHDEST"].ne(""), "DISCHDEST"].unique())
    out["dq.mort30.sameday_disch_n"] = int((mort & _positive(a["SAMEDAY_DISCH"])).sum())
    out["dq.expired.mismatch_total_n"] = int(
        out["dq.expired.no_mort30_n"] + out["dq.mort30.dischdest_not_expired_n"])
    out["dq.expired.note"] = (
        "A patient recorded as having expired before discharge has no discharge day. These "
        "patients are the competing event in the Aalen-Johansen estimator and must never be "
        "read as same-day discharges. Both directions of the reconciliation are published: "
        "expired patients without a recorded death day, and recorded deaths whose discharge "
        "destination is not expired, which are the patients who died after discharge."
    )
    return out


# ===========================================================================
# 6. cross-checks the constitution asks 02 to report
# ===========================================================================
def cross_checks(d: pd.DataFrame, mask_analytic: pd.Series) -> dict:
    a = d[mask_analytic]
    both = a["DOPTODIS"].notna() & a["TOTHLOS"].notna() & a["HTOODAY"].notna()
    subtraction = a["TOTHLOS"] - a["HTOODAY"]
    disagree = both & (subtraction != a["DOPTODIS"])
    delta = (subtraction - a["DOPTODIS"])[disagree]

    # --- CLAUDE.md 9.4: is MORT30 really a 30-day flag? --------------------
    dop = a["DOPERTOD"]
    yonly = a["_YRDEATH_ONLY"]
    out_win = a["_YRDEATH_ONLY_OUT_OF_WINDOW"]
    expired = out_win & a["DISCHDEST"].eq("Expired")
    mort = {
        "derived.mort30.n": int(a["MORT30"].sum()),
        "derived.mort30.rate_pct": pct(int(a["MORT30"].sum()), len(a)),
        "derived.mort30.dopertod_min": float(dop.min()) if dop.notna().any() else float("nan"),
        "derived.mort30.dopertod_max": float(dop.max()) if dop.notna().any() else float("nan"),
        "derived.mort30.dopertod_outside_window_n": int(((dop < 0) | (dop > 30)).sum()),
        "derived.mort_any.n": int(a["MORT_ANY_RECORDED"].sum()),
        "derived.mort_any.rate_pct": pct(int(a["MORT_ANY_RECORDED"].sum()), len(a)),
        "derived.mort_any.yrdeath_n": int(a["YRDEATH"].notna().sum()),
        "derived.mort_any.yrdeath_only_n": int(yonly.sum()),
        "derived.mort_any.yrdeath_only_out_of_window_n": int(out_win.sum()),
        "derived.mort_any.yrdeath_only_out_of_window_expired_n": int(expired.sum()),
        "derived.mort_any.yrdeath_only_max_los_postop": float(a.loc[yonly, "DOPTODIS"].max()),
        "derived.mort_any.out_of_window_los_postop_days": sorted(
            int(v) for v in a.loc[out_win, "DOPTODIS"].dropna()),
        # Tied to the destination rather than to a list position, so the sentence in
        # notes_02.md cannot silently swap the two cases if the data ever changes.
        "derived.mort_any.out_of_window_expired_days": sorted(
            int(v) for v in a.loc[expired, "DOPTODIS"].dropna()),
        "derived.mort_any.out_of_window_discharged_alive_days": sorted(
            int(v) for v in a.loc[out_win & ~expired, "DOPTODIS"].dropna()),
    }

    # --- CLAUDE.md 9.6: the NONHOME_DISCH era break at 2021 ---------------
    # A SECONDARY OUTCOME, not a covariate, so the break matters more, not less. The rate
    # is computed on non-missing NONHOME_DISCH ('Other' destinations are missing, not 0).
    era = {}
    nh = a["NONHOME_DISCH"].astype("Float64")
    dest_facility = a["DEST_CAT"].eq("Facility")
    for label, sel in (("2012_2020", a["PUFYEAR"] <= 2020), ("2021_2024", a["PUFYEAR"] >= 2021)):
        sub, den = nh[sel], int(nh[sel].notna().sum())
        num = int((sub == 1).sum())
        era[f"derived.nonhome.era_{label}.cases"] = int(sel.sum())
        era[f"derived.nonhome.era_{label}.denominator"] = den
        era[f"derived.nonhome.era_{label}.n"] = num
        era[f"derived.nonhome.era_{label}.rate_pct"] = pct(num, den)
        era[f"derived.nonhome.era_{label}.dischdest_levels"] = int(
            a.loc[sel & a["DISCHDEST"].ne(""), "DISCHDEST"].nunique())
        era[f"derived.nonhome.era_{label}.facility_levels"] = int(
            a.loc[sel & dest_facility, "DISCHDEST"].nunique())
    era["derived.nonhome.overall.rate_pct"] = pct(int((nh == 1).sum()), int(nh.notna().sum()))
    era["derived.nonhome.era_break_relative_drop_pct"] = round(
        100.0 * (era["derived.nonhome.era_2012_2020.rate_pct"]
                 - era["derived.nonhome.era_2021_2024.rate_pct"])
        / era["derived.nonhome.era_2012_2020.rate_pct"], 2)
    era["derived.nonhome.no_trend_claim"] = True

    # --- CLAUDE.md 8.7: the length-of-stay field agreement, before either is used ------
    # `DOPTODIS` is the PRIMARY postoperative length-of-stay field: direct, present in every
    # study year, no arithmetic. `TOTHLOS - HTOODAY` is the INDEPENDENT cross-check and is
    # never the analysis variable. Every disagreement is a data-quality finding.
    los = {
        "losfield.doptodis_null": int(a["DOPTODIS"].isna().sum()),
        "losfield.tothlos_null": int(a["TOTHLOS"].isna().sum()),
        "losfield.htooday_null": int(a["HTOODAY"].isna().sum()),
        "losfield.both_present": int(both.sum()),
        "losfield.check_present": int(a["LOS_POSTOP_CHECK"].notna().sum()),
        "losfield.agree_n": int((a["LOS_POSTOP_AGREE"] == 1).sum()),
        "losfield.disagree_n": int(disagree.sum()),
        "losfield.disagree_pct_of_both_present": pct(int(disagree.sum()), int(both.sum())),
        "losfield.disagree_min_delta": float(delta.min()) if len(delta) else float("nan"),
        "losfield.disagree_max_delta": float(delta.max()) if len(delta) else float("nan"),
        "losfield.disagree_all_one_day_longer": bool(
            len(delta) > 0 and float(delta.min()) == 1.0 and float(delta.max()) == 1.0),
        "losfield.disagree_deltas": sorted(set(int(v) for v in delta.dropna())),
        "losfield.parent_disagree_n": CONSTITUTION_LOS_DISAGREEMENTS,
        "losfield.parent_both_present": CONSTITUTION_LOS_BOTH_PRESENT,
        "losfield.reproduces_parent": bool(
            int(disagree.sum()) == CONSTITUTION_LOS_DISAGREEMENTS
            and int(both.sum()) == CONSTITUTION_LOS_BOTH_PRESENT),
        "losfield.primary_field": "DOPTODIS",
        "losfield.crosscheck_field": "TOTHLOS minus HTOODAY",
        "losfield.doptodis_median": float(a["DOPTODIS"].median()),
        "losfield.doptodis_min": float(a["DOPTODIS"].min()),
        "losfield.doptodis_max": float(a["DOPTODIS"].max()),
        "losfield.doptodis_zero_days": int((a["DOPTODIS"] == 0).sum()),
        "losfield.doptodis_gt30_n": int((a["DOPTODIS"] > 30).sum()),
        "losfield.sameday_disch_n": int((a["SAMEDAY_DISCH"] == 1).sum()),
        "losfield.disch_by_pod1_n": int((a["DISCH_BY_POD1"] == 1).sum()),
    }
    for lvl in VD.LOS_CAT_ORDER:
        los[f"losfield.los_cat.{lvl.replace('>=', 'ge')}.n"] = int((a["LOS_CAT"] == lvl).sum())
    los["losfield.los_cat.missing_n"] = int((a["LOS_CAT"] == "").sum())

    # --- STILLINHOSP, described by measurement rather than by assertion ---
    # An earlier revision of this block published the claim that these patients "have no
    # post-discharge period at all". THAT IS FALSE for the ones who carry a length-of-stay
    # value, so the claim is replaced by the sub-counts and every one of them is a key a
    # downstream step can read instead of parsing prose.
    still = _still_in_hospital(a)
    los_val = a["LOS_POSTOP"]
    still_los = still & los_val.notna()
    postdisch_cols = [c for c in VD.POSTDISCH_COLUMNS if c in a.columns]
    postd_any = a[postdisch_cols].notna().any(axis=1)
    postd_post = (a[postdisch_cols] == 1).any(axis=1)
    postd_pre = (a[postdisch_cols] == 0).any(axis=1)
    los.update({
        "losfield.stillinhosp_yes_n": int(still.sum()),
        "losfield.stillinhosp_levels_observed": sorted(
            v for v in _text(a["STILLINHOSP"]).unique() if v),
        "losfield.stillinhosp_los_missing_n": int((still & los_val.isna()).sum()),
        "losfield.stillinhosp_los_present_n": int(still_los.sum()),
        "losfield.stillinhosp_los_gt30_n": int((still_los & (los_val > 30)).sum()),
        "losfield.stillinhosp_los_le30_n": int((still_los & (los_val <= 30)).sum()),
        "losfield.stillinhosp_los_le30_values": sorted(
            int(v) for v in los_val[still_los & (los_val <= 30)]),
        "losfield.stillinhosp_postdisch_observed_n": int((still & postd_any).sum()),
        "losfield.stillinhosp_postdisch_scored_post_n": int((still & postd_post).sum()),
        "losfield.stillinhosp_postdisch_scored_pre_n": int((still & postd_pre).sum()),
        "losfield.stillinhosp_sameday_disch_n": int(
            (still & _positive(a["SAMEDAY_DISCH"])).sum()),
        "losfield.stillinhosp_disch_by_pod1_n": int(
            (still & _positive(a["DISCH_BY_POD1"])).sum()),
        "losfield.los_gt30_n": int((los_val > 30).sum()),
        "losfield.los_gt30_stillinhosp_n": int(((los_val > 30) & still).sum()),
        "losfield.stillinhosp_note": (
            "The still-in-hospital flag means the patient was still admitted beyond 30 days "
            "after the operation. It does NOT mean the patient has no length of stay and it "
            "must not be described that way: most carry no length-of-stay value, but the "
            "rest do, some of them 30 days or fewer, and the ones whose event day and "
            "length of stay are both present are already inside 04's row-4 denominators, "
            "every one of them scored pre-discharge. The sub-counts are published as their "
            "own keys so a downstream step reads numbers rather than this sentence."),
        "losfield.stillinhosp_bounding_note": (
            "Binding on 04_postdischarge.py: these are the patients the bounding "
            "sensitivity must be conditioned on. A patient never discharged inside the "
            "follow-up window cannot have had an event after a discharge that did not "
            "occur, so the STILLINHOSP-conditioned bound is the primary one and the naive "
            "unconditional bound may be shown only beside it, labelled as such."),
    })

    # --- who is inside SAMEDAY_DISCH and DISCH_BY_POD1 but did not go home ---
    # The two columns are honest as "postoperative length of stay is 0" and "1 or less" and
    # their DEFINITIONS ARE NOT CHANGED: 05 needs the raw form. What was missing is the
    # contamination count, so a same-day-discharge RATE cannot be reported without the
    # reader knowing that a handful of those rows are deaths and patients never discharged.
    sameday = _positive(a["SAMEDAY_DISCH"])
    bypod1 = _positive(a["DISCH_BY_POD1"])
    mort30 = _positive(a["MORT30"])
    expired_dest = a["DISCHDEST"].eq("Expired")
    for label, sel in (("sameday_disch", sameday), ("disch_by_pod1", bypod1)):
        los[f"losfield.{label}_expired_dest_n"] = int((sel & expired_dest).sum())
        los[f"losfield.{label}_mort30_n"] = int((sel & mort30).sum())
        los[f"losfield.{label}_stillinhosp_n"] = int((sel & still).sum())
        los[f"losfield.{label}_not_discharged_home_any_n"] = int(
            (sel & (expired_dest | mort30 | still)).sum())
    los["losfield.discharge_contamination_note"] = (
        "SAMEDAY_DISCH and DISCH_BY_POD1 are defined on the postoperative length-of-stay "
        "field alone and their definitions are deliberately unchanged, because 05_inout.py "
        "needs the raw form for the agreement analysis. They therefore contain patients who "
        "did not go home: an expired discharge destination, a recorded death within 30 "
        "days, or a still-in-hospital flag. Binding on 05_inout.py: exclude them explicitly "
        "and state the exclusion, rather than reporting a same-day-discharge rate that "
        "counts deaths as discharges.")

    # --- the derived outcome columns, counted on the analytic cohort ------
    outcomes = {
        "derived.reop_trach.slot1_n": int(a["REOP_TRACH"].sum()),
        "derived.reop_any.n": int(a["REOP_ANY"].sum()),
        "derived.airway.reintubation_n": int(a["REINTUBATION"].sum()),
        "derived.airway.vent48_n": int(a["VENT48"].sum()),
        "derived.airway.composite_n": int(a["AIRWAY_COMP"].sum()),
        "derived.pneumonia.n": int(a["PNEUMONIA"].sum()),
        "derived.readmit_unplanned.n": int(a["READMIT_UNPLANNED"].sum()),
        "derived.window_comp.n": int(a["WINDOW_COMP"].sum()),
        "derived.window_comp.rate_pct": pct(int(a["WINDOW_COMP"].sum()), len(a)),
        "derived.window_comp.components": list(VD.WINDOW_COMPONENTS),
        "derived.window_comp.hematoma_n": int(a["HEMATOMA"].sum()),
        "derived.window_comp.pod_present_n": int(a["POD_WINDOW"].notna().sum()),
        "derived.window_comp.pod_missing_n": int(
            int(a["WINDOW_COMP"].sum()) - int(a["POD_WINDOW"].notna().sum())),
        "derived.window_comp.overlap_hematoma_and_airway_n": int(
            (_positive(a["HEMATOMA"]) & _positive(a["AIRWAY_COMP"])).sum()),
        "derived.dischdest.home_n": int((a["DEST_CAT"] == "Home").sum()),
        "derived.dischdest.facility_n": int((a["DEST_CAT"] == "Facility").sum()),
        "derived.dischdest.other_n": int((a["DEST_CAT"] == "Other").sum()),
        "derived.dischdest.missing_n": int((a["DEST_CAT"] == "").sum()),
        "derived.dischdest.facility_which_was_home_n": int(
            (a["DISCHDEST"] == "Facility Which was Home").sum()),
        "derived.dischdest.levels_observed_n": int(
            a.loc[a["DISCHDEST"].ne(""), "DISCHDEST"].nunique()),
    }

    # --- the billing label, on the analytic cohort ------------------------
    # `05_inout.py` owns the AGREEMENT ANALYSIS and the reconciliation of the carried-over
    # figure. What is published here is only the column build: the level counts and the
    # completeness of the recode, in this step's own namespace.
    inout = {
        "inoutfield.analytic_n": len(a),
        "inoutfield.outpatient_n": int((a["INOUT"] == VD.INOUT_OUTPATIENT_LEVEL).sum()),
        "inoutfield.inpatient_n": int((a["INOUT"] == VD.INOUT_INPATIENT_LEVEL).sum()),
        "inoutfield.blank_n": int((a["INOUT"] == "").sum()),
        "inoutfield.levels_observed": sorted(v for v in a["INOUT"].unique() if v),
        "inoutfield.flag_present_n": int(a["INOUT_OUTPT"].notna().sum()),
        "inoutfield.outpatient_pct": pct(
            int((a["INOUT"] == VD.INOUT_OUTPATIENT_LEVEL).sum()), len(a)),
        "inoutfield.is_billing_label": True,
        "inoutfield.note": (
            "The billing setting is not a discharge measure and is never used as one. True "
            "same-day discharge is derived from the postoperative length-of-stay field. "
            "05_inout.py owns the agreement analysis and the reconciliation of the "
            "carried-over outpatient count; these keys are the column build only."),
    }

    return {**mort, **era, **los, **outcomes, **inout}


# ===========================================================================
# 6. writers
# ===========================================================================
# (column, dtype-in-parquet, origin, definition) -- the source of cohort_schema.md.
SCHEMA_SPEC = [
    ("CASE_UID", "str", "raw", "Unique key, '<puf_year>-<CASEID>'. CASEID alone repeats across years."),
    ("PUFYEAR", "Int16", "raw", "Warehouse year (CLAUDE.md 9.2). The curated view COALESCEs the in-file 2012-2014 hole with puf_year, so this column is 0% NULL."),
    ("OPERYR", "Int16", "derived", "Set equal to PUFYEAR after asserting the two agree on every row. FEATURE_PRETTY names OPERYR; the era logic uses PUFYEAR; they can never drift."),
    ("AGE", "Int16", "raw", "Years, top-coded at 90 ('90+' in the raw PUF, SMALLINT 90 on the curated view)."),
    ("AGE_PER10", "float64", "derived", "AGE / 10. The age term of the both-ways comparison covariate set (CLAUDE.md 6.6); no other regression in this project reads it."),
    ("SEX", "str", "raw", "'male' / 'female' / 'non-binary'."),
    ("MALE", "Int8", "derived", "1 if SEX == 'male', 0 if 'female', MISSING for 'non-binary' (37 cases identified / 35 analytic). Non-binary is not female, so folding it into the reference group would be misclassification."),
    ("RACE_CAT", "str", "derived", "Six locked levels. ETHNICITY_HISPANIC == 'Yes' wins and maps to 'Hispanic'; otherwise RACE_NEW is mapped by an explicit table; comma-joined multi-race strings map to 'Other'; blank maps to 'Unknown'. An unmapped single-race level RAISES."),
    ("HEIGHT", "float64", "raw", "Inches."),
    ("WEIGHT", "float64", "raw", "Pounds."),
    ("BMI", "float64", "derived", f"703 * WEIGHT / HEIGHT^2. Set MISSING outside the CHOSEN clinical plausibility window [{BMI_MIN:.0f}, {BMI_MAX:.0f}] kg/m2. It is a chosen clinical rule and must never be described as discovered. PORTED UNCHANGED from the companion project, because moving a bound here would fork the two cohorts. The affected count and its hematoma events are published under derived.bmi.* and listed in cohort_schema.md under the decisions this file made."),
    ("BMI_PER5", "float64", "derived", "BMI / 5. The body-mass-index term of the both-ways comparison covariate set (CLAUDE.md 6.6)."),
    ("ASACLAS", "str", "raw", "Four levels survive the cohort definition; '5-Moribund' and 'None assigned' are excluded."),
    ("ASA_CAT", "str", "derived", "'1-2' or '>=3' from the leading digit of ASACLAS."),
    ("ASA_GE3", "Int8", "derived", "1 if the leading digit of ASACLAS is 3 or 4. The ASA term of the both-ways comparison covariate set, and the same information as the locked ASA_CAT stratifier."),
    ("FNSTATUS2", "str", "raw", "Functional status. FNSTATUS1 is 100% NULL database-wide and is not exported."),
    ("FNSTATUS_DEPENDENT", "Int8", "derived", "1 for Partially or Totally Dependent, 0 for Independent, MISSING for 'Unknown' (a real level, 851 identified, rising sharply since 2022)."),
    ("DIABETES", "str", "raw", "UPPERCASE levels NO / NON-INSULIN / INSULIN."),
    ("DIABETES_ANY", "Int8", "derived", "1 if insulin or non-insulin diabetes."),
    ("BLEEDDIS", "Int8", "recoded", "Bleeding disorder, Yes/No text recoded in place to 0/1. NOT a drug exposure (CLAUDE.md 7.1). Component 1 of the coagulopathy burden."),
    ("SMOKE", "Int8", "recoded", "Current smoker, Yes/No recoded in place to 0/1."),
    ("HXCOPD", "Int8", "recoded", "COPD, Yes/No recoded in place to 0/1."),
    ("HXCHF", "Int8", "recoded", "Heart failure, Yes/No recoded in place to 0/1. Prevalence TRIPLES at 2021 despite a documented 'terminology only' change (CLAUDE.md 6.7); do not pool without the caveat."),
    ("HYPERMED", "Int8", "recoded", "Hypertension requiring medication, Yes/No recoded in place to 0/1. PORTED; it is a member of the completeness roster only and no regression in this project reads it."),
    ("DIALYSIS", "Int8", "recoded", "Dialysis dependence, Yes/No recoded in place to 0/1."),
    ("STEROID", "Int8", "recoded", "Chronic steroid or immunosuppressant, Yes/No recoded in place to 0/1. The definition step is at 2021, not 2019. EXCLUDED from every analysis and every table in this project (CLAUDE.md 6.7); the column exists only because the ported in-place recode roster names it and dropping it would fork the two cohorts."),
    ("PRHCT", "float64", "raw", "Preoperative hematocrit, %."),
    ("PRALBUM", "float64", "raw", "Preoperative albumin, g/dL."),
    ("PRINR", "float64", "raw", "Preoperative INR."),
    ("PRPLATE", "float64", "raw", "Preoperative platelet count, x10^3/uL."),
    ("PRPTT", "float64", "raw", "Preoperative partial thromboplastin time, seconds."),
    ("ANEMIA", "Int8", "derived", "Preoperative anemia, WHO hematocrit thresholds: PRHCT < 39 in men, < 36 in women. MISSING when PRHCT is missing or sex is non-binary (the definition is sex-specific)."),
    ("INR_GT12", "Int8", "derived", "PRINR > 1.2. MISSING when PRINR is missing."),
    ("PLT_LT150", "Int8", "derived", "PRPLATE < 150. MISSING when PRPLATE is missing."),
    ("PTT_GT35", "Int8", "derived", "PRPTT > 35. MISSING when PRPTT is missing."),
    ("COAG_BURDEN_N", "Int8", "derived", "Count of {BLEEDDIS, INR_GT12, PLT_LT150, PTT_GT35}, 0-4. Defined ONLY where LAB_COMPLETE; otherwise MISSING, because scoring an unmeasured lab as normal would dilute the exposure."),
    ("COAG_BURDEN_PER1", "float64", "derived", "The same count as a float, for the ordinal model term. Numerically identical to COAG_BURDEN_N."),
    ("COAG_BURDEN_CAT", "str", "derived", "'0' / '1' / '>=2'; empty string where COAG_BURDEN_N is missing."),
    ("COAG_BURDEN_GE2", "Int8", "derived", "1 if COAG_BURDEN_N >= 2. The threshold exposure."),
    ("LEVELS_N", "Int16", "derived", f"cpt_acdf.count_levels over OTHERCPT1..10 = 1 + occurrences of CPT {CPT.ADDITIONAL_LEVEL_CPT}. UNCAPPED, so the right tail runs past 3. CONCPT* is NOT scanned."),
    ("LEVELS_PER_LEVEL", "float64", "derived", "LEVELS_N as a float, UNCAPPED. PORTED. THIS PROJECT FITS NEITHER PER-LEVEL TERM: it fits no hematoma risk model at all (CLAUDE.md 6.6). Both are materialised so the two cohorts carry the same columns."),
    ("LEVELS_PER_LEVEL_CAP3", "float64", "derived", f"LEVELS_N CAPPED at {LEVELS_CAP} ({LEVELS_CAP} or more levels all score {LEVELS_CAP}), as a float, for the per-level model term. The cap is EXPLICIT and load-bearing: an estimate from this column would be 'per level up to {LEVELS_CAP} or more', never 'per level'. PORTED so the two cohorts carry the same columns; THIS PROJECT FITS NO MODEL ON IT (CLAUDE.md 6.6)."),
    ("LEVELS_CAT", "str", "derived", "'1' / '2' / '>=3' from cpt_acdf.level_stratum."),
    ("LEVELS_GE3", "Int8", "derived", f"1 if LEVELS_N >= {LEVELS_CAP}."),
    ("ANT_INSTRUM", "Int8", "derived", f"Anterior cervical instrumentation: CPT {cpt_list(CPT.ANTERIOR_INSTRUMENTATION_CPTS)} in OTHERCPT1..10."),
    ("OPTIME", "float64", "raw", "Operative time, minutes. 12 analytic cases missing."),
    ("SURGSPEC", "str", "raw", "Twelve levels in this cohort."),
    ("SPEC_CAT", "str", "derived", "'Neurosurgery' / 'Orthopaedic' / 'Other'. Note the raw level is spelled 'Orthopedics'."),
    ("ELECTIVE", "Int8", "derived", "NULL-safe elective flag: EMERGNCY != 'Yes' AND (CASETYPE is NULL OR CASETYPE == 'Elective'). EMERGNCY stops at 2020 and CASETYPE starts at 2021, so a predicate on either alone silently drops the other era."),
    ("EXCL_CORPECTOMY", "Int8", "derived", f"CPT {cpt_list(CPT.EXCLUSION_CORPECTOMY_CPTS)} in OTHERCPT1..10. Figure 1 exclusion box."),
    ("EXCL_POSTERIOR", "Int8", "derived", f"CPT {cpt_list(CPT.EXCLUSION_POSTERIOR_CERVICAL_CPTS)} in OTHERCPT1..10. Describe as 'concurrent posterior spinal decompression or instrumentation', never 'posterior cervical' (the list carries a lumbar code and codes that are not cervical-specific; CLAUDE.md 9.17)."),
    ("EXCL_ARTHROPLASTY", "Int8", "derived", f"CPT {cpt_list(CPT.EXCLUSION_ARTHROPLASTY_CPTS)} in OTHERCPT1..10."),
    ("EXCL_ASA5", "Int8", "derived", "ASACLAS == '5-Moribund'."),
    ("EXCL_ASA_UNASSIGNED", "Int8", "derived", "ASACLAS == 'None assigned'. An UNDOCUMENTED exclusion in the plan prose that the plan's own numbers require; CLAUDE.md 9.15 makes it its own STROBE box."),
    ("EXCL_PRESEPSIS", "Int8", "derived", "PRSEPIS != 'None'. The literal string 'None' means NO sepsis."),
    ("STEP_ELECTIVE", "Int8", "derived", "Reached the elective rung of the ladder."),
    ("STEP_EXCLUSIONS", "Int8", "derived", "Reached the post-procedure-exclusion rung."),
    ("STEP_ANALYTIC", "Int8", "derived", "In the analytic cohort. Always 1 in cohort_analytic.parquet."),
    ("HEMATOMA", "Int8", "derived", f"PRIMARY OUTCOME. Union of reoperation CPT, ICD-9 and ICD-10 arms, reoperation SLOT 1 only, UNGATED across all years. Seroma codes excluded. The ICD-10 code set is cpt_acdf.HEMATOMA_DEFINITION = {CPT.HEMATOMA_DEFINITION} ({CPT.HEMATOMA_ICD10_VARIANTS[CPT.HEMATOMA_DEFINITION]['label']}); see cpt_acdf section 5a for why the original set was corrected and what each variant measures."),
    # CLAUDE.md hard rule 5: not one code literal, including in a description string. The
    # differences between the three pre-specified code sets are RENDERED from cpt_acdf, so a
    # code can only ever be typed once, in the module that owns the taxonomy.
    ("HEMATOMA_V0", "Int8", "derived", f"SENSITIVITY. The same union under the ORIGINAL ICD-10 code set. Relative to the primary set it CARRIED {icd_list(_variant_extra('V0'), ', ') or 'no additional codes'} (one of which is not an ICD-10-CM code in any fiscal year and the other is accidental puncture and laceration, not a bleeding code), it LACKED the postprocedural-hemorrhage arm {icd_list(_variant_missing('V0'), ', ') or 'none'}, and its seroma guard did not hold {icd_list(_variant_guard_gap('V0'), ', ') or 'any additional code'}. Published as a named sensitivity row, never as the primary outcome."),
    ("HEMATOMA_V1", "Int8", "derived", f"SENSITIVITY. Error corrections only: the two non-bleeding codes removed and the two nervous-system seroma codes added to the guard, but WITHOUT the postprocedural-hemorrhage arm {icd_list(_variant_missing('V1'), ', ') or 'none'} that the primary definition adds."),
    ("HEMATOMA_CPT_ONLY", "Int8", "derived", f"Arm 1 alone: {CPT.HEMATOMA_CPT_COLUMN} in {{{cpt_list(CPT.HEMATOMA_REOP_CPTS, ', ')}}}. For the strict-definition sensitivity."),
    ("HEMATOMA_ICD_ONLY", "Int8", "derived", f"Arms 2 and 3: {CPT.HEMATOMA_ICD9_COLUMN} or {CPT.HEMATOMA_ICD10_COLUMN}. For the strict-ICD sensitivity."),
    ("HEMATOMA_POD", "float64", "derived", "RETORPODAYS where HEMATOMA == 1, else MISSING, with the timing-field handling rule applied. This is the postoperative-day column the timed-outcome roster points the hematoma outcome at; POD_HEMATOMA does not exist and must not be created."),
    ("REOP_ANY", "Int8", "derived", "REOPERATION1 == 'Yes'. Asserted identical to RETURNOR."),
    ("REINTUBATION", "Int8", "derived", "REINTUB != 'No Complication'. Testing == 'Yes' yields all zeros."),
    ("VENT48", "Int8", "derived", "FAILWEAN != 'No Complication'."),
    ("PNEUMONIA", "Int8", "derived", "OUPNEUMO != 'No Complication'. EXCLUDED from AIRWAY_COMP."),
    ("REOP_TRACH", "Int8", "derived", f"CPT {CPT.TRACHEOSTOMY_CPT} in {CPT.HEMATOMA_CPT_COLUMN} (slot 1 only, the companion project's slot ruling). Exploratory marker with no literature precedent (the companion project's exploratory-marker ruling)."),
    ("AIRWAY_COMP", "Int8", "derived", "CLAUDE.md 6.2: REINTUBATION OR VENT48 OR REOP_TRACH. Pneumonia is deliberately EXCLUDED."),
    ("READMIT_UNPLANNED", "Int8", "derived", "UNPLANNEDREADMISSION1 == 'Yes'. That column is populated only when READMISSION1 == 'Yes', so NULL means no readmission, not 'not unplanned'."),
    ("MORT30", "Int8", "derived", "PRIMARY 30-day mortality. There is NO DEATH column (CLAUDE.md 9.4). 1 if DOPERTOD (days from operation to death) is present. DOPERTOD is VERIFIED at build time to lie within [0, 30] and the build STOPS if it ever does not, so this column really is a 30-day flag. CLAUDE.md 9.4."),
    ("MORT_ANY_RECORDED", "Int8", "derived", "SENSITIVITY ONLY, and NEVER to be called 30-day mortality. 1 if DOPERTOD is present OR YRDEATH is present. YRDEATH is a YEAR and is NOT window-bounded: among the cases it alone flags, one died on postoperative day 33 during the index stay and one was discharged ALIVE on day 41. Report it beside MORT30 and state the difference (CLAUDE.md 9.4)."),
    ("LOS_POSTOP", "float64", "derived", "DOPTODIS, days from operation to discharge = TRUE postoperative length of stay, and the PRIMARY length-of-stay field for this project. It is NOT range-limited to 30 days: a stay longer than the follow-up window is real and truncating it would corrupt the discharge-timing analysis. The billing setting INOUT is never used as a length of stay."),
    ("DISCHDEST", "str", "raw", "Fourteen levels observed in this cohort; the vocabulary swaps completely at 2021."),
    ("DEST_CAT", "str", "derived", "'Home' / 'Facility' / 'Other' from variable_dict.DISCHDEST_LEVEL_MAP, an explicit map covering BOTH vocabularies and published as supplementary table S1. 'Facility Which was Home' maps to Facility, NOT Home, and substring-matching the word Home would get that wrong. An unmapped level RAISES; a blank returns the empty string."),
    ("NONHOME_DISCH", "Int8", "derived", "1 if DEST_CAT == 'Facility', 0 if 'Home', MISSING for 'Other' (against medical advice, expired, unknown) and for a missing destination. HAS A 2021 ERA BREAK and carries NO trend claim (CLAUDE.md 9.6): the rate steps down as DISCHDEST collapses from seven facility levels to two. Measured per-era rates are in notes_02.md section 5 and in results_02.json under nonhome.era_*."),
    ("DYSPHAGIA_READMIT", "Int8", "derived", f"{icd_list(CPT.DYSPHAGIA_ICD10_PREFIXES)} family prefix match on {CPT.DYSPHAGIA_READMISSION_ICD10_COLUMN} (slot 1). ICD-10 only, so the denominator is 2014-2024."),
    ("DYSPHAGIA_REOP", "Int8", "derived", f"{icd_list(CPT.DYSPHAGIA_ICD10_PREFIXES)} family prefix match on {CPT.DYSPHAGIA_REOPERATION_ICD10_COLUMN} (slot 1)."),
    ("DYSPHAGIA_ANY", "Int8", "derived", "DYSPHAGIA_READMIT OR DYSPHAGIA_REOP."),
    ("MODEL_COMPLETE", "Int8", "derived", f"PORTED COMPLETENESS FLAG, NOT A MODEL. Complete on all {len(VD.MODEL_COMPLETE_TERMS)} columns of variable_dict.MODEL_COMPLETE_TERMS ({', '.join(VD.MODEL_COMPLETE_TERMS)}). The parent computed it from a constant named PRIMARY_MODEL_TERMS, which is deleted here because this project fits no hematoma risk model; the roster is the same six columns in the same order, so the flag is byte-identical and the two cohorts do not fork. Read it only to reproduce that flag; fit nothing on it."),
    ("LAB_COMPLETE", "Int8", "derived", "PRINR, PRPLATE and PRPTT all measured. PRPTT is the binding constraint, not INR. PORTED; never read by this manuscript."),
    ("INR_MEASURED", "Int8", "derived", "PRINR measured. PORTED; never read by this manuscript."),
    # ---------------- NEW IN THIS PROJECT (.build_facts.md 5.2) ----------------
    ("INOUT", "str", "raw", "BILLING setting, two levels, Inpatient and Outpatient. It is NOT a discharge measure and never a length of stay: confusing the two is the measurement error this paper documents. Mapped through an explicit table in variable_dict; an unmapped level RAISES and the recode is NULL-safe."),
    ("INOUT_OUTPT", "Int8", "derived", "1 where the billing setting is outpatient, 0 where inpatient, MISSING otherwise. From variable_dict.inout_flag."),
    ("LOS_POSTOP_CHECK", "float64", "derived", "TOTHLOS minus HTOODAY, the INDEPENDENT cross-check on postoperative length of stay (CLAUDE.md 8.7). NEVER the analysis variable; LOS_POSTOP is."),
    ("LOS_POSTOP_AGREE", "Int8", "derived", "1 where LOS_POSTOP equals LOS_POSTOP_CHECK, 0 where both are present and they differ, MISSING where either is missing."),
    ("LOS_CAT", "str", "derived", "'0' / '1' / '>=2' postoperative nights from LOS_POSTOP; the EMPTY STRING where LOS_POSTOP is missing. Table 1 stratum. Note that pretty_level returns an empty string rather than a missing-value label for the blank, so a step displaying this column must route the missing stratum explicitly."),
    ("SAMEDAY_DISCH", "Int8", "derived", "1 where postoperative length of stay is 0 days, that is TRUE same-day discharge. MISSING where the length of stay is missing. This is the trend definition, and it is derived from the length-of-stay field and never from the billing setting. It is a LENGTH-OF-STAY statement, not a destination statement: a small number of these patients did not go home, having an expired discharge destination, a recorded death within 30 days, or a still-in-hospital flag. The definition is deliberately unchanged because 05_inout.py needs the raw form; the contamination counts are published under losfield.sameday_disch_* and 05 must exclude them explicitly and state the exclusion."),
    ("DISCH_BY_POD1", "Int8", "derived", "1 where postoperative length of stay is 1 day or less. MISSING where the length of stay is missing. The same caveat as SAMEDAY_DISCH applies and the counts are published under losfield.disch_by_pod1_*."),
    ("AGE_CAT", "str", "derived", "Locked age stratifier, levels '18-49' / '50-64' / '65-74' / '>=75', taken from variable_dict.AGE_CAT_ORDER. Cut points fixed a priori before any curve was fitted."),
    ("REOP_TRACH_SLOT12", "Int8", "derived", f"CPT {CPT.TRACHEOSTOMY_CPT} in {CPT.HEMATOMA_CPT_COLUMN} OR in {REOP_SLOT2_CPT_COLUMN}. ASCERTAINMENT SUPPLEMENT ONLY and deliberately NOT in AIRWAY_COMP or WINDOW_COMP, so the composite stays the parent's frozen slot-1 implementation and the two papers cannot disagree about it. The build re-asserts every run that the composite is identical either way."),
    ("WINDOW_COMP", "Int8", "derived", "The safe-discharge composite, HEMATOMA OR AIRWAY_COMP, which is hematoma requiring unplanned reoperation OR unplanned reintubation OR ventilator beyond 48 hours OR reoperative tracheostomy. Pneumonia and unplanned readmission are EXCLUDED and their absence is load-bearing. It is not one of the eight timed ladder outcomes and therefore has no post-discharge or both-fields-present companion column."),
    ("POD_REOP_ANY", "float64", "derived", "RETORPODAYS where REOP_ANY == 1, else MISSING, with the timing-field handling rule applied."),
    ("POD_REINTUB", "float64", "derived", "DREINTUB where REINTUBATION == 1, else MISSING, with the timing-field handling rule applied."),
    ("POD_VENT48", "float64", "derived", "DFAILWEAN where VENT48 == 1, else MISSING, with the timing-field handling rule applied."),
    ("POD_PNEUMONIA", "float64", "derived", "DOUPNEUMO where PNEUMONIA == 1, else MISSING, with the timing-field handling rule applied."),
    ("POD_READMIT", "float64", "derived", "READMPODAYS1 where READMIT_UNPLANNED == 1, else MISSING, with the timing-field handling rule applied. Days are counted from the principal operation, so a readmission is a post-discharge event BY CONSTRUCTION for anyone who was discharged; the curve SHAPE is reported and the post-discharge fraction is not."),
    ("POD_DEATH", "float64", "derived", "DOPERTOD where MORT30 == 1, else MISSING. MORT30 is defined as DOPERTOD being present, so this column is populated for every death in the window."),
    ("POD_TRACH", "float64", "derived", "The reoperation day for the slot-1 reoperative-tracheostomy arm of the airway composite."),
    ("POD_TRACH12", "float64", "derived", "The EARLIEST reoperation day across slots 1 and 2 for REOP_TRACH_SLOT12. Ascertainment supplement only."),
    ("POD_AIRWAY", "float64", "derived", "The EARLIEST observed day among the airway-composite components that actually fired, taken from the reintubation day, the prolonged-ventilation day and the reoperation day for the tracheostomy arm. MISSING where the composite fired but no component day was observed; it is never imputed and never given a boundary value."),
    ("POD_WINDOW", "float64", "derived", "The EARLIEST observed day among the WINDOW_COMP components that actually fired, that is the earlier of the hematoma reoperation day and POD_AIRWAY."),
    ("ERA_ICD10", "Int8", "derived", "1 where the warehouse operative year is 2016 or later. The coding-era sensitivity stratum."),
    ("ERA_EMERGNCY", "Int8", "derived", "1 where the warehouse operative year is 2020 or earlier, that is the era in which the elective definition came from the emergency-case field rather than the case-type field."),
]

# The two generated timing families, one row per timed outcome, expanded from the ROSTER so
# the schema cannot name a column the builder does not produce (or miss one it does).
for _o in VD.TIMED_OUTCOMES:
    SCHEMA_SPEC.append((
        VD.postdisch_column(_o.key), "Int8", "derived",
        f"1 where the postoperative day of {_o.noun} EXCEEDS that patient's own "
        f"postoperative length of stay, 0 where it does not, MISSING where either side is "
        f"missing or the event did not fire. LOCKED TIE CONVENTION: an event on the same "
        f"postoperative day as discharge counts as PRE-discharge, which is the conservative "
        f"choice for a safety claim. The opposite convention is a required sensitivity and "
        f"is computed inline by 04_postdischarge.py from these frozen columns; it is "
        f"deliberately not materialised here."))
for _o in VD.TIMED_OUTCOMES:
    SCHEMA_SPEC.append((
        VD.timed_both_column(_o.key), "Int8", "derived",
        f"Row 4 of the post-discharge denominator ladder for {_o.noun}: 1 where the event "
        f"fired AND both {_o.pod_col} and LOS_POSTOP are present, 0 where the event fired "
        f"and at least one is missing, MISSING where the event did not fire. Summing it "
        f"gives row 4 directly; counting its zeros gives the events lost between rows 1 "
        f"and 4."))


def write_schema_md(d_analytic: pd.DataFrame, results: dict) -> None:
    lines = [
        "# cohort_schema.md -- the frozen analytic dataset",
        "",
        "Written by `02_build_cohort.py`. **`derived/cohort_analytic.parquet` is FROZEN "
        "after this step.** Every downstream step reads it and never re-derives a cohort "
        "(`CLAUDE.md` 4.1). If a column you need is not here, that is a bug in `02`, not a "
        "licence to compute it locally.",
        "",
        f"* `derived/cohort_analytic.parquet` -- **n = {results['cohort.analytic.n']:,}**, "
        f"{results['cohort.analytic.events']:,} hematoma events "
        f"({results['cohort.analytic.rate_pct']:.3f}%), "
        f"{results['derived.schema.n_columns']} columns.",
        f"* `derived/cohort_identified.parquet` -- n = "
        f"{results['cohort.identified.n']:,}, pre-exclusion, for Figure 1 only.",
        "",
        "Missing is `pd.NA` / `NaN` throughout. Binary columns are nullable `Int8`, so "
        "\"missing\" and \"no\" are distinguishable; never `fillna(0)` a binary without "
        "saying so.",
        "",
        "## Column dictionary",
        "",
        "`origin` is `raw` (as exported, only whitespace-stripped or numerically coerced), "
        "`recoded` (an NSQIP column whose values were replaced in place) or `derived` "
        "(a new column).",
        "",
        "| Column | dtype | origin | definition |",
        "|---|---|---|---|",
    ]
    for name, dtype, origin, definition in SCHEMA_SPEC:
        if name not in d_analytic.columns:
            raise SystemExit(f"[02] cohort_schema.md lists {name}, which was not created")
        lines.append(f"| `{name}` | {dtype} | {origin} | {definition} |")

    documented = {row[0] for row in SCHEMA_SPEC}
    passthrough = [c for c in d_analytic.columns if c not in documented]
    lines += [
        "",
        "## Raw passthrough columns",
        "",
        f"{len(passthrough)} further columns arrive from the export bundle unchanged "
        "(whitespace-stripped; the date/day/lab columns numerically coerced) and are "
        "carried so a reviewer request cannot force a re-export. They are NOT part of the "
        "plan's analysis and must not appear in a table unless asked for.",
        "",
        "```",
    ]
    for i in range(0, len(passthrough), 6):
        lines.append("  " + "  ".join(f"{c:<24}" for c in passthrough[i:i + 6]).rstrip())
    lines += [
        "```",
        "",
        "## Decisions this file MADE rather than inherited",
        "",
        "Each of these was under-specified by the plan and by every handoff note. They are "
        "listed so a reviewer can challenge them, and repeated in `derived/notes_02.md`.",
        "",
        f"1. **THE TIMING-FIELD HANDLING RULE.** {TIMING_RANGE_RULE} Measured on the "
        f"analytic cohort, **{results['timingfield.total_dropped_by_range_rule_n']} values "
        f"across all {len(TIMING_FIELDS)} days-to-event fields** were affected. Every field "
        f"was checked for non-integer and out-of-range values and the counts are published "
        f"under `timingfield.*`. The rule is applied ONCE, here, and every downstream step "
        f"inherits it through the frozen `POD_*` columns; **no step clips or drops "
        f"locally.** It does NOT apply to the length-of-stay fields, which are not "
        f"window-bounded: `LOS_POSTOP` runs to "
        f"{results['losfield.doptodis_max']:.0f} days and truncating it would corrupt the "
        f"discharge-timing analysis.",
        f"2. **The tie convention is LOCKED** (`CLAUDE.md` 6.3): "
        f"`POSTDISCH = (POD > LOS_POSTOP)`, so an event on the same postoperative day as "
        f"discharge counts as PRE-discharge. Only this convention is materialised; the "
        f"opposite one is a required sensitivity and `04_postdischarge.py` computes it "
        f"inline from the frozen columns, which `CLAUDE.md` 4.1 grants as its single "
        f"explicit exception.",
        f"3. **Every `POD_*` column is masked by its own event flag.** A days-to-event value "
        f"present on a row whose event flag is negative is counted and published under "
        f"`dq.negflag_days.*` and never used.",
        f"4. **`TIMED_BOTH_*` is defined conditional on the event**: 1 where the event fired "
        f"and both fields are present, 0 where the event fired and one is missing, MISSING "
        f"where the event did not fire. Its sum is row 4 of the denominator ladder and its "
        f"zero count is the loss between rows 1 and 4.",
        f"5. **Composite timing is the EARLIEST observed day among the components that "
        f"fired**, and a composite that fired with no observed component day stays MISSING. "
        f"It is never imputed and never given a boundary value.",
        f"6. **BMI plausibility window [{BMI_MIN:.0f}, {BMI_MAX:.0f}] kg/m2.** Outside it "
        f"`BMI` is missing; {results['derived.bmi.implausible_analytic_n']} analytic cases "
        f"are affected and they carry "
        f"{results['derived.bmi.implausible_analytic_events']} hematoma events. It is a "
        f"CHOSEN clinical rule, PORTED UNCHANGED from the companion project so the two "
        f"cohorts cannot fork. It must never be described as discovered.",
        "7. **`MALE` is missing for `SEX == 'non-binary'`** rather than folded into the "
        "female reference group.",
        "8. **`ANEMIA` uses the WHO sex-specific hematocrit thresholds** (39% men, 36% "
        "women), so it is undefined for non-binary sex.",
        f"9. **`MORT30` is `DOPERTOD`-based ({results['derived.mort30.n']}), not the "
        f"`DOPERTOD` OR `YRDEATH` union ({results['derived.mort_any.n']})**, per "
        f"`CLAUDE.md` 9.4. `DOPERTOD` is verified every run to lie within [0, 30]; "
        f"`YRDEATH` is a year and is not window-bounded. The union is materialised as "
        f"`MORT_ANY_RECORDED` and is a SENSITIVITY column that is never called 30-day "
        f"mortality.",
        f"10. **Both fusion-level per-level terms are materialised and NEITHER is fitted "
        f"here**: `LEVELS_PER_LEVEL` uncapped and `LEVELS_PER_LEVEL_CAP3` capped at "
        f"{LEVELS_CAP}. This project fits no hematoma risk model at all (`CLAUDE.md` 6.6).",
        "11. **`COAG_BURDEN_N` is defined only where `LAB_COMPLETE`.** An unmeasured lab is "
        "not a normal lab. PORTED; never read by this manuscript.",
        "12. **`NONHOME_DISCH` is missing, not 0, for `DEST_CAT == 'Other'`** (against "
        "medical advice, expired, unknown), and it carries NO trend claim because of the "
        "2021 vocabulary break.",
        "13. **Seven raw Yes/No comorbidity columns are recoded in place to 0/1** "
        f"({', '.join('`' + c + '`' for c in BINARY_INPLACE)}), because "
        "`variable_dict.FEATURE_PRETTY` declares each a model feature and none has a "
        "derived twin. Every other raw column keeps its NSQIP text.",
        f"14. **`AGE_CAT` cut points** are `{'`, `'.join(VD.AGE_CAT_ORDER)}`, chosen a "
        f"priori and locked before any curve was fitted (`CLAUDE.md` 6.5).",
        f"15. **The three airway and pneumonia days-to-event fields are numerically "
        f"coerced here**, where the parent left them as passthrough text: this project "
        f"times all eight outcomes and the parent timed none.",
        f"16. **The loss between denominator-ladder rows 1 and 4 is DECOMPOSED, not just "
        f"counted**, into a missing days-to-event field, a missing length of stay on a "
        f"patient flagged still in hospital beyond 30 days, and a missing length of stay on "
        f"a patient not so flagged. No count is reclassified and no rung moves. The split is "
        f"published under `timingfield.<outcome>.lost_*` because a patient never discharged "
        f"inside the follow-up window cannot have had the event after a discharge that did "
        f"not occur, which makes `PLAN.md` 6.2's unconditional extreme structurally "
        f"impossible for those rows. **`04_postdischarge.py` reports the "
        f"`STILLINHOSP`-conditioned bound as primary** and may show the unconditional one "
        f"only beside it, labelled as such.",
        "",
    ]
    SCHEMA_MD.write_text("\n".join(lines) + "\n")


def _notes_header(r: dict) -> str:
    hem_def = r["cohort.hematoma.definition"]
    return f"""# notes_02.md -- handoff from `02_build_cohort.py`

Written 2026-07-31. Owner: **T4 (cohort)**. Owned files: `02_build_cohort.py`,
`derived/cohort_analytic.parquet`, `derived/cohort_identified.parquet`,
`derived/cohort_schema.md`, `derived/notes_02.md`, `outputs/results_02.json`. Nothing else
was created or modified; the warehouse was not read at all (this step reads the export
bundle `01_export_cohort.py` already wrote).

**Read `derived/cohort_schema.md` first.** It is the column dictionary. This file is the
part `03_timing.py`, `04_postdischarge.py`, `05_inout.py` and `06_window.py` have to act on.

This builder is a PORT of `projects/ACDF-hematoma/v1/02_build_cohort.py`, copied first and
then extended, so the cohort ladder logic arrived byte-identical and cannot have broken
through transcription error. Everything added is a new derived column or a new measurement.

---

## 0. The three gates

| Gate | What it claims | Verdict |
|---|---|---|
| **B** (default, PRIMARY) | the corrected cohort ladder | **{'PASS' if r['gate.b.pass'] else 'FAIL'}** |
| **A** (`--replicate-plan`) | the plan's strict age cast, reconciliation only | **{'PASS' if r['gate.a.pass'] else 'FAIL'}** |
| **1** (CLAUDE.md 8.3) | adding the billing column moved no row count | **{'PASS' if r['gate.1.pass'] else 'FAIL'}** |

| Step | Gate B, PRIMARY | events | Gate A, reconciliation | events |
|---|---:|---:|---:|---:|
| identified | {r['cohort.identified.n']:,} | {r['cohort.identified.events']} | {r['gate.a.identified.n']:,} | {r['gate.a.identified.events']} |
| elective | {r['cohort.elective.n']:,} | {r['cohort.elective.events']} | {r['gate.a.elective.n']:,} | {r['gate.a.elective.events']} |
| minus procedure exclusions | {r['cohort.exclusions.n']:,} | {r['cohort.exclusions.events']} | {r['gate.a.exclusions.n']:,} | {r['gate.a.exclusions.events']} |
| **analytic** | **{r['cohort.analytic.n']:,}** | **{r['cohort.analytic.events']}** | **{r['gate.a.analytic.n']:,}** | **{r['gate.a.analytic.events']}** |

Every one of those sixteen counts is asserted against `cpt_acdf.COHORT_LADDER_N`,
`COHORT_LADDER_N_PLAN` and `COHORT_LADDER_EVENTS`. **No count is retyped anywhere in this
step.** The event ladder follows `cpt_acdf.HEMATOMA_DEFINITION`, which is
**{hem_def}**; `PLAN.md` prints the superseded V0 ladder for the reason `CLAUDE.md`
section 0 documents, and importing lands on the current definition automatically. Analytic
hematoma rate {r['cohort.analytic.rate_pct']:.3f}% on Gate B. The event counts are identical
on both ladders because the {r['cohort.nonagenarian_delta.identified']} restored
top-coded-age patients carry zero hematoma events.

`--replicate-plan` **writes nothing**. It asserts, prints and exits, so the reconciliation
mode can never clobber a frozen artefact. Its output belongs to the ascertainment
supplement and never to the manuscript body.

**Gate 1**, in four legs, all of which held:

1. **Width.** {r['gate.1.columns']} = {r['gate.1.parent_columns']} parent columns plus
   {r['gate.1.added_columns']}.
2. **Structural, ladder level.** What was checked:
   {r['gate.1.structural_postderive_detail']}. On its own this leg proves only that the
   ladder functions do not read the added column, because the eligibility masks are already
   materialised by the time it runs. It is reported separately for exactly that reason.
3. **Structural, full rebuild from the raw CSV.** What was checked:
   {r['gate.1.structural_rebuild_detail']}. The bundle is re-read from disk, the added column
   is deleted before any derivation touches it, and both ladders are rebuilt through the real
   derivation path, which reads the added column in one block only (its own column build,
   skipped in this mode). Any other read, at any depth, raises rather than quietly returning
   a different mask. The rebuild reproduced
   {r['gate.1.rebuild.b.identified.n']:,} to {r['gate.1.rebuild.b.elective.n']:,} to
   {r['gate.1.rebuild.b.exclusions.n']:,} to {r['gate.1.rebuild.b.analytic.n']:,} with events
   {r['gate.1.rebuild.b.identified.events']} to {r['gate.1.rebuild.b.elective.events']} to
   {r['gate.1.rebuild.b.exclusions.events']} to {r['gate.1.rebuild.b.analytic.events']}, and
   the Gate A ladder likewise.
4. **Numeric.** The ladder landed on the pinned constants with
   **{r['gate.1.rows_moved']} rows moved**.

T1 had already proved the stronger export-layer claim: {r['gate.1.export_proof']}

STROBE boxes for Figure 1, measured on the Gate B ladder:

| Exclusion | n removed |
|---|---:|
| not elective | {r['cohort.flow.not_elective.n']:,} |
| concurrent corpectomy or craniocervical arthrodesis | {r['cohort.flow.excl_corpectomy.n']:,} |
| concurrent posterior spinal decompression or instrumentation | {r['cohort.flow.excl_posterior.n']:,} |
| cervical disc arthroplasty | {r['cohort.flow.excl_arthroplasty.n']:,} |
| ASA class 5 | {r['cohort.flow.excl_asa5.n']:,} |
| ASA not assigned | {r['cohort.flow.excl_asa_unassigned.n']:,} |
| preoperative sepsis | {r['cohort.flow.excl_presepsis.n']:,} |

The three procedure boxes overlap (a case can carry two exclusion codes) and remove
{r['cohort.flow.excl_procedure_any.n']:,} cases between them. **The final rung's arithmetic
does not close, and it must be stated rather than adjusted** (`CLAUDE.md` 9.16): the rung
difference is {r['cohort.flow.exclusion_delta']}, the three final boxes sum to
{r['cohort.flow.exclusion_boxes_sum']}, so **{r['cohort.flow.exclusion_overlap_n']}** case
meets more than one exclusion criterion. Do not adjust a count to make it close. The
ASA-not-assigned box is an exclusion the plan's prose does not mention and its own numbers
require; it gets its own box with its own n.

**Binding on `07` and `08`: build Figure 1 and the Methods exclusion sentence from
`outputs/results_02.json` (`cohort.flow.*`), never from a handoff note.** Every box above is
rendered from that fragment, and this project already has two recorded instances of a count
being copied wrong into a report while the artefact was right. The keys are
`cohort.flow.excl_asa5.n`, `cohort.flow.excl_asa_unassigned.n`,
`cohort.flow.excl_presepsis.n`, `cohort.flow.exclusion_delta` and
`cohort.flow.exclusion_overlap_n`.

---

## 1. THE PRE-FLIGHT DATA CHECK (`CLAUDE.md` 8.6), before any curve is drawn

### 1.1 The complete observed value set of every timing field

**Finding: within the analytic cohort every days-to-event field is a clean integer in the
range 0 to 30. There is nothing to clip and nothing to drop.**

The warehouse catalog reports 41 distinct values for the reoperation-day field and 48 for
the readmission-day field **across all of `main`**, where a clean 0 to 30 integer range
would give 31. Those counts are database-wide across every surgical specialty and every
year. Measured inside this cohort, no field exceeds 31 distinct values and none carries a
value outside the window:

| Field | present | distinct | min | max | non-integer | outside 0-30 | dropped by the rule |
|---|---:|---:|---:|---:|---:|---:|---:|
"""


def _notes_timing_tables(r: dict) -> str:
    rows = []
    for f in TIMING_FIELDS:
        k = f.lower()
        lo = r[f"timingfield.{k}.min"]
        hi = r[f"timingfield.{k}.max"]
        lo_s = "n/a" if lo != lo else f"{lo:.0f}"
        hi_s = "n/a" if hi != hi else f"{hi:.0f}"
        rows.append(
            f"| `{f}` | {r[f'timingfield.{k}.present_n']:,} | "
            f"{r[f'timingfield.{k}.distinct_n']} | {lo_s} | {hi_s} | "
            f"{r[f'timingfield.{k}.noninteger_n']} | "
            f"{r[f'timingfield.{k}.outside_0_30_n']} | "
            f"{r[f'timingfield.{k}.dropped_by_range_rule_n']} |")
    los_rows = []
    for f in LOS_FIELDS:
        k = f.lower()
        los_rows.append(
            f"| `{f}` | {r[f'timingfield.{k}.present_n']:,} | "
            f"{r[f'timingfield.{k}.distinct_n']} | {r[f'timingfield.{k}.min']:.0f} | "
            f"{r[f'timingfield.{k}.max']:.0f} | "
            f"{r[f'timingfield.{k}.outside_0_30_n']:,} |")
    absent = {f: r[f"timingfield.{f.lower()}.values_absent_from_0_30"] for f in TIMING_FIELDS}
    gaps = "; ".join(
        f"`{f}` is missing {v}" for f, v in absent.items() if v and len(v) <= 6) or "none"
    ladder = []
    for k in VD.TIMED_OUTCOME_KEYS:
        o = VD.timed_outcome(k)
        ladder.append(
            f"| {VD.short(o.event_col)} | `{o.event_col}` | "
            f"{r[f'timingfield.{k}.events']:,} | {r[f'timingfield.{k}.pod_present_n']:,} | "
            f"{r[f'timingfield.{k}.pod_missing_n']:,} | "
            f"{r[f'timingfield.{k}.pod_missing_pct']:.2f}% | "
            f"{r[f'timingfield.{k}.los_present_n']:,} | "
            f"{r[f'timingfield.{k}.both_present_n']:,} | "
            f"{r[f'timingfield.{k}.lost_rows_1_to_4_pct']:.2f}% | "
            f"{'YES' if r[f'timingfield.{k}.bounding_required'] else 'no'} |")
    decomp = []
    for k in VD.TIMED_OUTCOME_KEYS:
        o = VD.timed_outcome(k)
        lost = r[f"timingfield.{k}.lost_rows_1_to_4_n"]
        los_lost = r[f"timingfield.{k}.lost_los_missing_n"]
        all_still = ("n/a" if los_lost == 0
                     else ("**yes**" if r[f"timingfield.{k}.lost_los_missing_all_stillinhosp"]
                           else "**NO**"))
        decomp.append(
            f"| {VD.short(o.event_col)} | {lost:,} | "
            f"{r[f'timingfield.{k}.lost_day_missing_n']:,} | "
            f"{r[f'timingfield.{k}.lost_los_missing_stillinhosp_n']:,} | "
            f"{r[f'timingfield.{k}.lost_los_missing_not_stillinhosp_n']:,} | "
            f"{all_still} | "
            f"{r[f'timingfield.{k}.bound_unconditional_max_n']:,} | "
            f"{r[f'timingfield.{k}.bound_stillinhosp_conditioned_max_n']:,} |")
    negflag = []
    for k in VD.TIMED_OUTCOME_KEYS:
        o = VD.timed_outcome(k)
        n = r[f"dq.negflag_days.{k}.n"]
        if n == 0:
            status, why = "none", "no days-to-event value is present on a negative row"
        elif r[f"dq.negflag_days.{k}.expected"]:
            status, why = "expected", r[f"dq.negflag_days.{k}.reason"]
        else:
            status, why = "UNEXPECTED", r[f"dq.negflag_days.{k}.reason"]
        negflag.append(
            f"| `{o.event_col}` | {', '.join('`' + f + '`' for f in o.raw_fields)} | "
            f"{n:,} | {status} | {why} |")
    strict = []
    for f in STRICT_GATE:
        k = f.lower()
        strict.append(
            f"| `{f}` | `{r[f'dq.negflag_days.strict.{k}.gate']}` | "
            f"{r[f'dq.negflag_days.strict.{k}.gate_positive_n']:,} | "
            f"{r[f'dq.negflag_days.strict.{k}.gate_positive_days_missing_n']:,} | "
            f"{r[f'dq.negflag_days.strict.{k}.n']:,} |")
    return "\n".join(rows) + f"""

Length-of-stay fields, audited the same way but **deliberately not range-limited**:

| Field | present | distinct | min | max | above 30 days |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(los_rows) + f"""

Gaps inside the 0 to 30 range (a day on which no event of that type happened to occur, not
a defect): {gaps}.

### 1.2 The handling rule, decided here and inherited everywhere

> **{r['timingfield.rule']}**

Applied ONCE, in `02`, through a single helper, and every downstream step inherits it
through the frozen `POD_*` columns. **No step clips or drops locally.** On this cohort the
rule affected **{r['timingfield.total_dropped_by_range_rule_n']} values in total**, which is
the honest answer: the rule exists because the data could have been dirty, it was checked,
and it is not. It is stated rather than assumed so that a future PUF vintage carrying a
malformed day cannot pass through silently.

**The rule does NOT apply to the length-of-stay fields.** `LOS_POSTOP` runs to
{r['losfield.doptodis_max']:.0f} days and {r['losfield.doptodis_gt30_n']:,} analytic
patients stayed longer than the 30-day follow-up window. A length of stay is not a
window-bounded quantity and truncating one would corrupt the discharge-timing analysis this
project exists to do.

### 1.3 Conditional missingness: the number that matters

For every timed outcome, the missingness of the days-to-event field **given that the event
flag is positive**. The database-wide percentages in `PLAN.md` 2.4 are EVENT RATES, not
missingness rates, and the manuscript must not confuse the two.

| Outcome | flag | events (row 1) | day present (row 2) | day missing | % missing given the event | length of stay present (row 3) | both (row 4) | lost rows 1 to 4 | bounding needed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(ladder) + f"""

Rows 1 to 4 are the `PLAN.md` 6.2 denominator ladder. `04_postdischarge.py` owns the
published version of this table; the copy here is `02`'s own measurement and is a
cross-check, not a duplicate. **Bounding is required for any outcome losing more than
{r['timingfield.bounding_threshold_pct']:.0f}% of its events between rows 1 and 4**
(`PLAN.md` 6.2), and the final column above says which.

### 1.4 WHY each event is lost between rows 1 and 4, and what `04` may bound with

The loss above is decomposed into three disjoint reasons that sum to it. **Nothing is
reclassified and no ladder count moves**; rows 1 through 4 are exactly as measured above.
This is the breakdown `04` needs to bound honestly.

| Outcome | lost rows 1 to 4 | day missing | length of stay missing, still in hospital | length of stay missing, NOT still in hospital | all length-of-stay loss still in hospital | naive bound may move | conditioned bound may move |
|---|---:|---:|---:|---:|---|---:|---:|
""" + "\n".join(decomp) + f"""

**Finding: every event lost because the LENGTH OF STAY is missing belongs to a patient
flagged still in hospital beyond 30 days.** The "not still in hospital" column is zero on
every outcome, which is what
`timingfield.loss_all_los_loss_is_stillinhosp` =
{str(r['timingfield.loss_all_los_loss_is_stillinhosp']).upper()} records.

Read the columns, not a slogan. **The three outcomes that trip the
{r['timingfield.bounding_threshold_pct']:.0f}% rule** (unplanned reintubation, ventilator
beyond 48 hours, the airway composite) **lose their events to that arm and to nothing else**,
so for them the whole row-1-to-4 loss is still-in-hospital and the conditioned bound has
zero width. Hematoma, any unplanned reoperation and pneumonia additionally lose
{r['timingfield.hematoma.lost_day_missing_n']},
{r['timingfield.reop_any.lost_day_missing_n']} and
{r['timingfield.pneumonia.lost_day_missing_n']} of their events to a missing event DAY, and
unplanned readmission loses all {r['timingfield.readmit.lost_day_missing_n']} of its events
that way. A missing day is not a still-in-hospital fact, so those events stay inside both
bounds; that is why the conditioned bound is not zero for those four outcomes.

**Why it matters, and it is binding on `04`.** The still-in-hospital flag means the patient
was still admitted beyond postoperative day 30, so the patient was never discharged inside
the follow-up window at all. An event on such a row **cannot** have occurred after that
patient's own discharge, because there was no discharge. `PLAN.md` 6.2's "all missing events
post-discharge" extreme is therefore not merely conservative for these rows, it is
**structurally impossible**, and printing the naive interval would make it materially wider
than the data supports. `PLAN.md` 6.2 also says an interval wide enough to change the
clinical message replaces the point estimate, so a spuriously wide one does real harm.

> **`04` must report the `STILLINHOSP`-conditioned bound as PRIMARY, and may show the naive
> unconditional bound only beside it, explicitly labelled as the unconditional extreme.**

The two right-hand columns are the counts each bound may move: the naive one is the whole
loss, the conditioned one is the loss that could genuinely be post-discharge, namely the
missing-day arm plus any length-of-stay loss on a patient not flagged still in hospital.
Every number here is published under `timingfield.<outcome>.lost_*` and
`timingfield.<outcome>.bound_*`; **re-derive rather than copy** (`CLAUDE.md` 8.5).

### 1.5 Days-to-event values on rows whose event flag is NEGATIVE

Counted, reported, and never used: every `POD_*` column is masked by its own event flag.

| Event flag | raw field(s) | days present with the flag negative | status | why |
|---|---|---:|---|---|
""" + "\n".join(negflag) + f"""

The same check against each field's **strict gate**, which is the flag whose positivity is
supposed to be necessary and sufficient for the field to be populated:

| Raw field | strict gate | gate positive | gate positive, day MISSING | day present, gate negative |
|---|---|---:|---:|---:|
""" + "\n".join(strict) + f"""

Of the {r['dq.negflag_days.strict.readmpodays1.n']:,} readmission days present without an
unplanned-readmission flag, **{r['dq.negflag_days.strict.readmpodays1.planned_readmission_n']:,}
carry a readmission flag** and are therefore planned readmissions, and
{r['dq.negflag_days.strict.readmpodays1.no_readmission_flag_n']:,} carry no readmission flag
at all. Total across outcomes with no documented explanation:
**{r['dq.negflag_days.unexplained_total_n']}**.
"""


def _notes_checks(r: dict) -> str:
    los_ok = "REPRODUCED" if r["losfield.reproduces_parent"] else "NOT REPRODUCED"
    same_delta = ("every one of them exactly one day longer"
                  if r["losfield.disagree_all_one_day_longer"]
                  else f"deltas {r['losfield.disagree_deltas']}")
    trach = "HOLDS" if r["derived.trach.invariant_holds"] else "BROKEN"
    out_days = " and ".join(
        str(v) for v in r["derived.mort_any.out_of_window_los_postop_days"])
    exp_days = " and ".join(str(v) for v in r["derived.mort_any.out_of_window_expired_days"])
    alive_days = " and ".join(
        str(v) for v in r["derived.mort_any.out_of_window_discharged_alive_days"])
    return f"""
---

## 2. Length-of-stay field agreement (`CLAUDE.md` 8.7), before either field is used

`LOS_POSTOP` is sourced from `DOPTODIS`: direct, present in every study year, no arithmetic.
`TOTHLOS - HTOODAY` is the independent cross-check and is materialised as
`LOS_POSTOP_CHECK`. **It is never the analysis variable.**

| Quantity | n |
|---|---:|
| `DOPTODIS` missing | {r['losfield.doptodis_null']:,} |
| `TOTHLOS` missing | {r['losfield.tothlos_null']:,} |
| `HTOODAY` missing | {r['losfield.htooday_null']:,} |
| all three present | {r['losfield.both_present']:,} |
| the two readings AGREE | {r['losfield.agree_n']:,} |
| the two readings DISAGREE | **{r['losfield.disagree_n']:,}** |

Disagreement rate {r['losfield.disagree_pct_of_both_present']:.3f}% of the rows where both
are computable, {same_delta} (minimum delta
{r['losfield.disagree_min_delta']:.0f}, maximum {r['losfield.disagree_max_delta']:.0f}): the
subtraction reads longer than the direct field. `CLAUDE.md` 8.7 records the parent's
measurement as {r['losfield.parent_disagree_n']} disagreements on
{r['losfield.parent_both_present']:,} analytic rows where both are present. Re-verified
here: **{los_ok}**.

Distribution of the primary field on the analytic cohort: median
{r['losfield.doptodis_median']:.0f} day, range {r['losfield.doptodis_min']:.0f} to
{r['losfield.doptodis_max']:.0f}, **{r['losfield.sameday_disch_n']:,} true same-day
discharges** ({pct(r['losfield.sameday_disch_n'], r['cohort.analytic.n']):.2f}%) and
{r['losfield.disch_by_pod1_n']:,} discharged by postoperative day 1
({pct(r['losfield.disch_by_pod1_n'], r['cohort.analytic.n']):.2f}%).
`LOS_CAT` is 0 nights {r['losfield.los_cat.0.n']:,}, 1 night {r['losfield.los_cat.1.n']:,},
2 or more nights {r['losfield.los_cat.ge2.n']:,}, missing
{r['losfield.los_cat.missing_n']:,}.

### 2.1 The still-in-hospital patients, described by measurement

**{r['losfield.stillinhosp_yes_n']:,} analytic patients are flagged still in hospital beyond
30 days after the operation.** An earlier revision of this file said they "have no
post-discharge period at all". **That claim is false for
{r['losfield.stillinhosp_los_present_n']} of them** and has been replaced by the counts
below, each of which is its own fragment key so a downstream step reads a number rather than
a sentence.

| Quantity | n |
|---|---:|
| flagged still in hospital beyond 30 days | **{r['losfield.stillinhosp_yes_n']:,}** |
| of those, `LOS_POSTOP` missing | {r['losfield.stillinhosp_los_missing_n']} |
| of those, `LOS_POSTOP` PRESENT | **{r['losfield.stillinhosp_los_present_n']}** |
| of those, length of stay above 30 days | {r['losfield.stillinhosp_los_gt30_n']} |
| of those, length of stay 30 days or fewer | **{r['losfield.stillinhosp_los_le30_n']}** |
| carrying a non-missing `POSTDISCH_*` value, so already inside `04`'s row-4 denominators | **{r['losfield.stillinhosp_postdisch_observed_n']}** |
| of those, scored PRE-discharge | {r['losfield.stillinhosp_postdisch_scored_pre_n']} |
| of those, scored POST-discharge | {r['losfield.stillinhosp_postdisch_scored_post_n']} |
| counted in `SAMEDAY_DISCH` | {r['losfield.stillinhosp_sameday_disch_n']} |
| counted in `DISCH_BY_POD1` | {r['losfield.stillinhosp_disch_by_pod1_n']} |

The lengths of stay of 30 days or fewer are
{', '.join(str(v) for v in r['losfield.stillinhosp_los_le30_values'])}. Conversely, of the
{r['losfield.los_gt30_n']} analytic patients whose stay exceeded 30 days only
{r['losfield.los_gt30_stillinhosp_n']} carry the flag, so the flag and a long stay are not
the same population in either direction.

**This does not weaken the bounding argument in section 1.4, it is what makes it
measurable.** The flag says the patient was still admitted beyond day 30, so no discharge
occurred inside the window; the patients who also carry a length-of-stay value are simply
the ones whose event day and length of stay are both recorded, and all
{r['losfield.stillinhosp_postdisch_scored_pre_n']} of the observed post-discharge scores are
pre-discharge.

### 2.2 `SAMEDAY_DISCH` and `DISCH_BY_POD1` include patients who did not go home

**The two column definitions are deliberately unchanged**: they are honest as "postoperative
length of stay is 0" and "1 day or less", and `05_inout.py` needs the raw form for the
agreement analysis. What is published here is the contamination, so no step can report a
same-day-discharge RATE without knowing what is inside it.

| Column | n | expired destination | recorded death (`MORT30`) | still in hospital | any of the three |
|---|---:|---:|---:|---:|---:|
| `SAMEDAY_DISCH` = 1 | {r['losfield.sameday_disch_n']:,} | {r['losfield.sameday_disch_expired_dest_n']} | {r['losfield.sameday_disch_mort30_n']} | {r['losfield.sameday_disch_stillinhosp_n']} | **{r['losfield.sameday_disch_not_discharged_home_any_n']}** |
| `DISCH_BY_POD1` = 1 | {r['losfield.disch_by_pod1_n']:,} | {r['losfield.disch_by_pod1_expired_dest_n']} | {r['losfield.disch_by_pod1_mort30_n']} | {r['losfield.disch_by_pod1_stillinhosp_n']} | **{r['losfield.disch_by_pod1_not_discharged_home_any_n']}** |

> **Binding on `05_inout.py`: exclude these patients explicitly and say so**, rather than
> reporting a same-day-discharge rate that counts deaths as discharges. A death on
> postoperative day 0 is not a discharge, and a patient still admitted beyond day 30 has not
> been discharged at all. The counts are keys under `losfield.*`; re-derive them rather than
> copying the table above.

---

## 3. `DISCHDEST = 'Expired'` reconciled against the death day (`CLAUDE.md` 8.8)

A patient who expired has no discharge day. They are the COMPETING EVENT in the estimator,
not a same-day discharge, and a length-of-stay field read naively would fold them into the
discharge distribution.

| Quantity | n |
|---|---:|
| discharge destination recorded as expired | **{r['dq.expired.n']:,}** |
| of those, a death day is recorded (`MORT30`) | {r['dq.expired.mort30_n']:,} |
| of those, NO death day is recorded | **{r['dq.expired.no_mort30_n']:,}** |
| of those, no death recorded by either field | {r['dq.expired.no_mort_any_recorded_n']:,} |
| expired patients carrying a length-of-stay value | {r['dq.expired.los_present_n']:,} |
| expired patients with length of stay 0 (would read as same-day discharge) | **{r['dq.expired.los_zero_n']:,}** |
| expired patients flagged still in hospital | {r['dq.expired.stillinhosp_yes_n']:,} |
| death day AFTER the recorded length of stay | {r['dq.expired.death_day_after_los_n']:,} |
| death day EQUAL to the recorded length of stay | {r['dq.expired.death_day_equals_los_n']:,} |
| death day BEFORE the recorded length of stay | {r['dq.expired.death_day_before_los_n']:,} |
| deaths in the window (`MORT30`) | {r['dq.mort30.n']:,} |
| of those, destination is expired | {r['dq.mort30.dischdest_expired_n']:,} |
| of those, destination is NOT expired (died after discharge) | **{r['dq.mort30.dischdest_not_expired_n']:,}** |
| of those, destination is home | {r['dq.mort30.dischdest_home_n']:,} |
| of those, destination is a facility | {r['dq.mort30.dischdest_facility_n']:,} |
| of those, destination is missing | {r['dq.mort30.dischdest_missing_n']:,} |
| of those, flagged as a same-day discharge | {r['dq.mort30.sameday_disch_n']:,} |

Destinations recorded for patients who died in the window:
{', '.join('`' + v + '`' for v in r['dq.mort30.dischdest_levels'])}.

**Both directions of the mismatch are real and neither is a coding bug.** A patient whose
destination is expired but who carries no death day is an ascertainment gap in the death
field. A patient with a recorded death day whose destination is not expired died AFTER
discharge, which is exactly what a 30-day follow-up window is supposed to capture and is
the reason death is the competing event rather than an administrative censoring. **Binding
on `03` and `06`: expired patients must enter the estimator as the competing event and must
never be counted as discharges.** Binding on `05`: any same-day-discharge count must
exclude them, and the count above is how many there are.

---

## 4. Second events, only partially timed (`CLAUDE.md` 9.18)

All analyses are **time-to-first-event**, pre-specified. NSQIP times occurrence 1 in every
family, plus a second reoperation and a second through fifth readmission, and nothing
beyond.

| Second event | flagged | day recorded |
|---|---:|---:|
| second unplanned reoperation | {r['dq.second_event.reop2.flag_n']:,} | {r['dq.second_event.reop2.days_present_n']:,} |
| third reoperation | {r['dq.second_event.reop3.flag_n']:,} | no timing field exists |
| second readmission | {r['dq.second_event.readmit2.flag_n']:,} | {r['dq.second_event.readmit2.days_present_n']:,} |
| third readmission | {r['dq.second_event.readmit3.flag_n']:,} | {r['dq.second_event.readmit3.days_present_n']:,} |
| fourth readmission | {r['dq.second_event.readmit4.flag_n']:,} | {r['dq.second_event.readmit4.days_present_n']:,} |
| fifth readmission | {r['dq.second_event.readmit5.flag_n']:,} | {r['dq.second_event.readmit5.days_present_n']:,} |

**{r['dq.second_event.any_n']:,} analytic patients ({r['dq.second_event.any_pct']:.2f}%) had
at least one second event**, and that is the figure the Methods must state as the population
whose timing is only partially observed.

---

## 5. The composite outcomes, and the tracheostomy invariant

Composite airway morbidity (`CLAUDE.md` 6.2), analytic cohort: unplanned reintubation
{r['derived.airway.reintubation_n']}, ventilator beyond 48 hours
{r['derived.airway.vent48_n']}, reoperative tracheostomy (slot 1)
{r['derived.reop_trach.slot1_n']}, **composite {r['derived.airway.composite_n']}**.
Pneumonia ({r['derived.pneumonia.n']}) is EXCLUDED and is reported as its own row.

The safe-discharge composite `WINDOW_COMP` = `{' OR '.join(VD.WINDOW_COMPONENTS)}`, which is
exactly hematoma OR the airway composite:

| Component | analytic n |
|---|---:|
| hematoma requiring unplanned reoperation | {r['derived.window_comp.hematoma_n']} |
| unplanned reintubation | {r['derived.airway.reintubation_n']} |
| ventilator beyond 48 hours | {r['derived.airway.vent48_n']} |
| reoperative tracheostomy, slot 1 | {r['derived.reop_trach.slot1_n']} |
| composite airway morbidity | {r['derived.airway.composite_n']} |
| **`WINDOW_COMP`** | **{r['derived.window_comp.n']}** ({r['derived.window_comp.rate_pct']:.3f}%) |
| patients meeting BOTH hematoma and the airway composite | {r['derived.window_comp.overlap_hematoma_and_airway_n']} |
| `POD_WINDOW` observed | {r['derived.window_comp.pod_present_n']} |
| `WINDOW_COMP` fired with no observed component day | {r['derived.window_comp.pod_missing_n']} |

**The tracheostomy invariant: {trach} on the analytic cohort**, which is the scope of the
claim and the only population any analysis sees. The composite is identical whether the
tracheostomy arm reads reoperation slot 1 only or slots 1 and 2:
**{r['derived.trach.invariant_rows_differing_analytic']} analytic rows differ.**

| Quantity | identified | analytic |
|---|---:|---:|
| reoperative tracheostomy, slot 1 | {r['derived.trach.slot1_identified_n']} | {r['derived.trach.slot1_analytic_n']} |
| reoperative tracheostomy, slots 1 and 2 | {r['derived.trach.slot12_identified_n']} | {r['derived.trach.slot12_analytic_n']} |
| slot-2-only cases the wider marker adds | {r['derived.trach.slot2_only_identified_n']} | {r['derived.trach.slot2_only_analytic_n']} |
| of those, already in the composite | {r['derived.trach.slot2_only_already_in_composite_n']} | {r['derived.trach.slot2_only_analytic_already_in_composite_n']} |
| composite rows that would change | {r['derived.trach.invariant_rows_differing_identified']} | **{r['derived.trach.invariant_rows_differing_analytic']}** |

All {r['derived.trach.slot2_only_analytic_n']} slot-2-only tracheostomies in the analytic
cohort already enter the composite through unplanned reintubation or prolonged ventilation,
which is exactly the companion project's finding, re-derived here rather than inherited.

**Published, not suppressed:** on the wider IDENTIFIED cohort the two readings differ on
{r['derived.trach.invariant_rows_differing_identified']} patient(s). Every one of them is
removed by the cohort definition before any analysis sees them, and the composite is only
ever computed on the analytic cohort, so this does not bear on any result. It is recorded
here because scoping a check to the population the claim is about is legitimate and hiding
the wider measurement would not be.

`REOP_TRACH_SLOT12` and `POD_TRACH12` are materialised for the ascertainment supplement and
are **not** in the composite, so this project's composite and the companion project's cannot
disagree.

---

## 6. Ported cross-checks the constitution asks `02` to report

**30-day mortality.** There is no death column in the warehouse, so mortality is built from
the death-day field and the death-year field. `MORT30` is the death-day flag and is verified
every run to lie inside the window; the union with the death year is `MORT_ANY_RECORDED`,
which is a SENSITIVITY column and **is never called 30-day mortality**.

| Quantity | n | rate |
|---|---:|---:|
| death day present, that is `MORT30`, PRIMARY | **{r['derived.mort30.n']}** | {r['derived.mort30.rate_pct']:.4f}% |
| death year present | {r['derived.mort_any.yrdeath_n']} | |
| union, that is `MORT_ANY_RECORDED`, SENSITIVITY | **{r['derived.mort_any.n']}** | {r['derived.mort_any.rate_pct']:.4f}% |
| cases the union adds | {r['derived.mort_any.yrdeath_only_n']} | |
| of those, provably outside the window | **{r['derived.mort_any.yrdeath_only_out_of_window_n']}** | |

The death day was verified to run {r['derived.mort30.dopertod_min']:.0f} to
{r['derived.mort30.dopertod_max']:.0f} days inclusive with
{r['derived.mort30.dopertod_outside_window_n']} values outside the window, and the build
STOPS if that ever fails. The death YEAR is not window-bounded: of the cases it alone flags,
those provably outside 30 days had postoperative stays of {out_days} days, of which
{r['derived.mort_any.yrdeath_only_out_of_window_expired_n']} died during the index stay on
day {exp_days} and the other was discharged ALIVE on day {alive_days}.

**Discharge destination.** Home {r['derived.dischdest.home_n']:,}, facility
{r['derived.dischdest.facility_n']:,}, other {r['derived.dischdest.other_n']:,}, missing
{r['derived.dischdest.missing_n']}, across
{r['derived.dischdest.levels_observed_n']} observed levels. The level map is imported from
`variable_dict.py` and an unmapped level RAISES. `Facility Which was Home`
({r['derived.dischdest.facility_which_was_home_n']} cases) is counted as **Facility**, not
Home; substring-matching the word Home would get that wrong.

**`NONHOME_DISCH` has a 2021 era break and carries NO trend claim** (`CLAUDE.md` 9.6):

| Era | cases | denominator | non-home | rate | destination levels | of which facility |
|---|---:|---:|---:|---:|---:|---:|
| 2012-2020 | {r['derived.nonhome.era_2012_2020.cases']:,} | {r['derived.nonhome.era_2012_2020.denominator']:,} | {r['derived.nonhome.era_2012_2020.n']:,} | **{r['derived.nonhome.era_2012_2020.rate_pct']:.2f}%** | {r['derived.nonhome.era_2012_2020.dischdest_levels']} | {r['derived.nonhome.era_2012_2020.facility_levels']} |
| 2021-2024 | {r['derived.nonhome.era_2021_2024.cases']:,} | {r['derived.nonhome.era_2021_2024.denominator']:,} | {r['derived.nonhome.era_2021_2024.n']:,} | **{r['derived.nonhome.era_2021_2024.rate_pct']:.2f}%** | {r['derived.nonhome.era_2021_2024.dischdest_levels']} | {r['derived.nonhome.era_2021_2024.facility_levels']} |

A {r['derived.nonhome.era_break_relative_drop_pct']:.0f}% relative drop, coinciding exactly
with the destination vocabulary collapsing from
{r['derived.nonhome.era_2012_2020.facility_levels']} facility levels to
{r['derived.nonhome.era_2021_2024.facility_levels']}. **Report destination by year as
`PLAN.md` 6.3 requires, but make no claim that non-home discharge declined.**

**The billing label on the analytic cohort.** Outpatient
{r['inoutfield.outpatient_n']:,} ({r['inoutfield.outpatient_pct']:.2f}%), inpatient
{r['inoutfield.inpatient_n']:,}, blank {r['inoutfield.blank_n']}. `05_inout.py` owns the
agreement analysis and the reconciliation of the carried-over outpatient count; these are
the column-build figures only, published under a different namespace so the two cannot
collide.

**Ascertainment across the coding transition.** Per-year analytic n and hematoma events are
in `results_02.json` under `cohort.analytic.by_year.*`. The year-by-year hematoma rate runs
{r['derived.rate_by_year.min_pct']:.3f}% to {r['derived.rate_by_year.max_pct']:.3f}% with no
structural break at the transition. **{r['derived.icd9_only.events_identified']} identified
events are ascertained by the ICD-9 arm alone**, running through
{r['derived.icd9_only.last_year']}, so a year-gated rule would lose the
{r['derived.icd9_only.events_2016']} coded in 2016. That is why the union is ungated.
"""


def _scale_family_count() -> int:
    """How many key families the nine declarations name, in total."""
    return sum(len(families) for _ns, _prose, families in SCALE_DECLARATIONS)


def _notes_handoff(r: dict) -> str:
    keys = ", ".join("`" + p.rstrip(".") + "`" for p in RESERVED_PREFIXES)
    return f"""
---

## 7. GATE 2 FOREWARNING: where the carried-over 41.3% comes from

`PLAN.md` 13 lists the denominator behind the carried-over "41.3% of hematoma reoperations
occur after that patient's own discharge" as an OPEN ITEM: the parent never recorded it.
The frozen columns answer it directly, so it is measured here rather than left for a later
step to rediscover. **`04_postdischarge.py` owns the published figure and must re-derive it;
these are `02`'s own measurements and are published under a data-quality namespace
precisely so they cannot be mistaken for the published value.**

| Rung | denominator (both fields present) | post-discharge n | % (primary tie) | % (opposite tie) |
|---|---:|---:|---:|---:|
| identified | {r['dq.carryover.postdisch.v2.identified.denominator']} | {r['dq.carryover.postdisch.v2.identified.post_n']} | **{r['dq.carryover.postdisch.v2.identified.pct']:.2f}%** | {r['dq.carryover.postdisch.v2.identified.pct_opposite_tie']:.2f}% |
| elective | {r['dq.carryover.postdisch.v2.elective.denominator']} | {r['dq.carryover.postdisch.v2.elective.post_n']} | {r['dq.carryover.postdisch.v2.elective.pct']:.2f}% | {r['dq.carryover.postdisch.v2.elective.pct_opposite_tie']:.2f}% |
| minus procedure exclusions | {r['dq.carryover.postdisch.v2.exclusions.denominator']} | {r['dq.carryover.postdisch.v2.exclusions.post_n']} | {r['dq.carryover.postdisch.v2.exclusions.pct']:.2f}% | {r['dq.carryover.postdisch.v2.exclusions.pct_opposite_tie']:.2f}% |
| **analytic (what the manuscript reports)** | **{r['dq.carryover.postdisch.v2.analytic.denominator']}** | **{r['dq.carryover.postdisch.v2.analytic.post_n']}** | **{r['dq.carryover.postdisch.v2.analytic.pct']:.2f}%** | {r['dq.carryover.postdisch.v2.analytic.pct_opposite_tie']:.2f}% |

**The carried-over 41.3% is an IDENTIFIED-cohort figure.** The closest reproduction across
all four rungs and both code sets is the **{r['dq.carryover.postdisch.closest_rung']}** rung
under the **{r['dq.carryover.postdisch.closest_definition'].upper()}** code set at
**{r['dq.carryover.postdisch.closest_pct']:.2f}%** on a denominator of
{r['dq.carryover.postdisch.closest_denominator']}, which rounds to 41.3% to one decimal:
reproduces = **{r['dq.carryover.postdisch.reproduces_to_one_decimal']}**. Under the
superseded code set the same rung gives
{r['dq.carryover.postdisch.v0.identified.pct']:.2f}%.

**Consequence, and it is a real one.** `CLAUDE.md` 4.4 rules that every analysis in this
manuscript is reported on the ANALYTIC cohort, so the published post-discharge fraction for
hematoma will be **{r['dq.carryover.postdisch.v2.analytic.pct']:.2f}%
({r['dq.carryover.postdisch.v2.analytic.post_n']} of
{r['dq.carryover.postdisch.v2.analytic.denominator']})**, about
{abs(r['dq.carryover.postdisch.v2.analytic.pct'] - 41.3):.1f} percentage points above the
carried-over figure. **That is not a discrepancy to reconcile away: it is the same measure
on a different and correctly stated denominator**, and the resolution `PLAN.md` 13 asked for
is exactly this. Report the analytic figure with its denominator, state that the
carried-over figure was an identified-cohort number, and cite the published external
estimate beside it rather than presenting either as confirming the other
(`CLAUDE.md` 6a.2).

The two companion carried-over figures behave the same way. Median postoperative day of the
hematoma reoperation is **{r['dq.carryover.pod01.v2.analytic.median']:.0f}** on the analytic
cohort and {r['dq.carryover.pod01.v2.identified.median']:.0f} on the identified cohort, so
the carried-over median of {r['dq.carryover.median_pod.plan_figure']:.0f} reproduces on
both. The share on postoperative day 0 or 1 is
{r['dq.carryover.pod01.v2.analytic.pct']:.2f}% analytic against
{r['dq.carryover.pod01.v2.identified.pct']:.2f}% identified, versus the carried-over
{r['dq.carryover.pod01.plan_figure_pct']:.1f}%. **`03_timing.py` owns the published median
and day 0 to 1 share; `04_postdischarge.py` owns the published post-discharge fraction.**
Neither may copy the numbers above; re-derive them and report a mismatch rather than
adopting or discarding one (`CLAUDE.md` 8.5).

---

## 8. What `03`, `04`, `05`, `06`, `07` and `08` must know

1. **Read the parquet, never the CSV, and never re-derive a cohort.**
   `derived/cohort_analytic.parquet` is FROZEN. Binary columns are nullable `Int8`; `NA`
   means missing, not "no". If a column you need is not there, that is a bug in `02`, not a
   licence to compute it locally.
2. **Iterate `variable_dict.TIMED_OUTCOMES`, never a local list.** Build every column name
   through `pod_column`, `postdisch_column` and `timed_both_column`. The hematoma outcome
   points at **`HEMATOMA_POD`**; **`POD_HEMATOMA` does not exist and must not be created**.
   The `POSTDISCH_` and `TIMED_BOTH_` suffixes come from the roster, not from the event
   column, so the reintubation columns are `POSTDISCH_REINTUB` and `TIMED_BOTH_REINTUB`
   while the flag is `REINTUBATION`.
3. **The timing-field handling rule is already applied.** Do not clip and do not drop. The
   `POD_*` columns are the post-rule values and the counts are published under
   `timingfield.*`. On this cohort the rule affected
   {r['timingfield.total_dropped_by_range_rule_n']} values.
4. **The tie convention is frozen in the `POSTDISCH_*` columns**:
   `POSTDISCH = (POD > LOS_POSTOP)`, an event on the discharge day counts as PRE-discharge.
   `04_postdischarge.py` alone may compute the opposite convention inline from `POD_*` and
   `LOS_POSTOP` (`CLAUDE.md` 4.1). Nothing else is exempt, and the opposite convention is
   deliberately not materialised so it cannot be used by accident.
5. **`TIMED_BOTH_*` is row 4 of the denominator ladder** and is defined conditional on the
   event: sum it for row 4, count its zeros for the loss between rows 1 and 4.
6. **THE BOUNDING SENSITIVITY MUST BE CONDITIONED ON `STILLINHOSP`. Binding on `04`.**
   Every event lost between rows 1 and 4 because the LENGTH OF STAY is missing belongs to a
   patient flagged still in hospital beyond 30 days, with zero exceptions on every outcome
   that loses any (`timingfield.loss_all_los_loss_is_stillinhosp` =
   {str(r['timingfield.loss_all_los_loss_is_stillinhosp']).upper()}), and such a patient was
   never discharged inside the window, so the event cannot be post-discharge. **Report the
   `STILLINHOSP`-conditioned bound as primary and show the naive unconditional bound only
   beside it, labelled as the unconditional extreme** (section 1.4). The per-outcome counts
   are `timingfield.<outcome>.lost_day_missing_n`, `.lost_los_missing_stillinhosp_n`,
   `.lost_los_missing_not_stillinhosp_n`, `.bound_unconditional_max_n` and
   `.bound_stillinhosp_conditioned_max_n`.
7. **`STILLINHOSP` does NOT mean "no length of stay".** It is false for
   {r['losfield.stillinhosp_los_present_n']} of the {r['losfield.stillinhosp_yes_n']}
   flagged patients, and {r['losfield.stillinhosp_postdisch_observed_n']} of them are already
   inside `04`'s row-4 denominators, all scored pre-discharge (section 2.1).
8. **Readmission has `postdisch_eligible = False`.** Report its curve SHAPE, never its
   post-discharge fraction: the day is counted from the operation, so a readmission is a
   post-discharge event by construction for anyone who was discharged.
9. **Expired patients are the competing event, never a same-day discharge** (section 3).
   {r['dq.expired.los_zero_n']} of them carry a length of stay of 0. More generally,
   **`SAMEDAY_DISCH` and `DISCH_BY_POD1` include patients who did not go home**:
   {r['losfield.sameday_disch_not_discharged_home_any_n']} and
   {r['losfield.disch_by_pod1_not_discharged_home_any_n']} respectively carry an expired
   destination, a recorded death or a still-in-hospital flag. The column definitions are
   deliberately unchanged; **`05` must exclude them explicitly and state the exclusion**
   (section 2.2).
10. **`LOS_CAT` writes the empty string where the length of stay is missing**
    ({r['losfield.los_cat.missing_n']} analytic cases), and `pretty_level` returns an empty
    string rather than a missing-value label for it. `05` must route that stratum explicitly
    or drop it with its count stated.
11. **`WINDOW_COMP` has `POD_WINDOW` and nothing else.** It is not one of the eight timed
    ladder outcomes and has no post-discharge or both-present column.
12. **The tracheostomy invariant is re-asserted on every run** and the composite stays on
    the frozen slot-1 implementation. `REOP_TRACH_SLOT12` and `POD_TRACH12` are the
    ascertainment-supplement companions; `03` reports the two counts side by side.
13. **Three things break at 2021** and any before-and-after sensitivity splits there, not at
    2019: the heart-failure comorbidity flag, the steroid flag (excluded from every table
    here anyway), and the discharge-destination vocabulary. `ERA_EMERGNCY` marks the
    2012-2020 era and `ERA_ICD10` the 2016-2024 one.
14. **`OPERYR` and `PUFYEAR` are the same column** by construction, both the warehouse year,
    asserted equal on every row. Use either; they cannot drift.
15. **The coagulopathy, dysphagia and laboratory columns are materialised but forbidden.**
    They exist so the two cohorts cannot fork (`CLAUDE.md` 7.1). No step here reads them,
    and `02` publishes no count from them.
16. **The pre-specification is published under `prespec.*`** so `verify_numbers.py` can
    assert nothing moved after the fact. Read the locked values from there rather than
    retyping them.
17. **Namespace discipline.** This fragment writes only {keys}. `los.` belongs to step 05,
    `era.` to step 03, and a duplicate key across fragments raises at the merge even when
    the values agree. To use a number another step measured, read it back from the merged
    `outputs/results.json`.
18. **Binding on `07` and `08`: build Figure 1 and the Methods exclusion sentence from
    `outputs/results_02.json` (`cohort.flow.*`), never from a handoff note**, including this
    one. Every count printed in section 0 above is rendered from that fragment. The exclusion
    boxes are {r['cohort.flow.excl_asa5.n']}, {r['cohort.flow.excl_asa_unassigned.n']} and
    {r['cohort.flow.excl_presepsis.n']}, summing to
    {r['cohort.flow.exclusion_boxes_sum']} against a rung difference of
    {r['cohort.flow.exclusion_delta']}, with an overlap of
    {r['cohort.flow.exclusion_overlap_n']}; read those five keys rather than these five
    numbers.
19. **Every one of this step's nine namespaces now declares its own units**
    (`.build_facts.md` 9.6, closed here). Each publishes `<namespace>.scale` as prose,
    `<namespace>.scale.families` and `<namespace>.scale.units` as two parallel
    machine-readable lists, and `<namespace>.scale.other_numeric_keys` as the catch-all, in
    the same shape `03`, `05` and `06` use. **{_scale_family_count()} families across {len(SCALE_DECLARATIONS)} namespaces,
    and the build ASSERTS that every published key matches exactly one of them**, so a gate
    can resolve the unit of any key in this fragment without a special case, and a new key
    added without a family fails inside `02`. Read the declaration belonging to the
    namespace that OWNS the key and never another step's: the generic interval suffixes
    resolve to at least four incompatible scales in the merged file. **Six families here
    carry a unit their name does not, and they are the ones to be careful with**:
    `derived.bmi.window_min` and `.window_max` are a body mass index,
    not days and not counts; `derived.mort30.dopertod_min` and `.dopertod_max` are
    postoperative days; `losfield.doptodis_zero_days` is a count of PATIENTS despite reading
    as a number of days; `dq.carryover.median_pod.plan_figure` and
    `dq.carryover.pod01.<version>.<rung>.median` are postoperative days published under a
    bare name; `derived.nonhome.era_break_relative_drop_pct` is a RELATIVE percentage of the
    earlier era's own rate and **not** a difference in percentage points; and
    `prespec.risk_thresholds` is held as PROPORTIONS on the 0 to 1 scale beside a list of
    rendered percentage labels.

---

## 9. Deviations from the ported parent builder

The cohort ladder logic is byte-identical to the parent's. What changed:

* **Kept and unchanged:** every eligibility predicate, the hematoma union and its two
  sensitivity variants, the BMI plausibility window, the anemia thresholds, the fusion-level
  terms, the coagulopathy columns, the dysphagia columns, the mortality construction, the
  discharge-destination collapse, and `MODEL_COMPLETE`.
* **`variable_dict.PRIMARY_MODEL_TERMS` no longer exists** and the builder now reads
  `MODEL_COMPLETE_TERMS`, the same six columns in the same order, so `MODEL_COMPLETE`
  reproduces byte for byte. The builder raises at import if the old constant ever comes
  back.
* **The discharge-destination level map is imported, not copied.** The parent carried its
  own literal; here `variable_dict.py` owns it and this file must not hold a second copy.
* **Added:** the billing-setting column and its flag, the length-of-stay cross-check and
  agreement flag, the length-of-stay and age strata, the two same-day-discharge flags, the
  safe-discharge composite, the wider tracheostomy marker, the two era flags, and the three
  timing families.
* **Dropped, with reason:** the parent's Gate D reconciliation and BMI-window sweep and its
  coagulopathy-burden variants (both target figures printed in the parent's own plan, which
  this project never quotes, and the burden rates are forbidden here in every form); the
  parent's consequence-cascade mortality block and its dysphagia counts (the parent's
  results, forbidden here); and the parent's seroma-guard reconstruction, which
  re-implements taxonomy internals that module's own self-test already covers. **None of
  these touches a cohort mask, and both ladders reproduce exactly.**
* **Three days-to-event fields are now numerically coerced** rather than carried as
  passthrough text, because this project times all eight outcomes and the parent timed none.
* **The fragment is re-namespaced.** The parent wrote 29 top-level prefixes; almost none is
  legal here, and every key is validated against this step's nine reserved prefixes before
  the file is written, so a namespace violation fails in `02` rather than at the merge.
"""


def write_notes_md(results: dict) -> None:
    text = (_notes_header(results) + _notes_timing_tables(results)
            + _notes_checks(results) + _notes_handoff(results))
    # CLAUDE.md hard rule 1. The constant is imported rather than written as a literal so a
    # repo-wide grep for U+2014 stays meaningful.
    if VD.EM_DASH in text:
        raise SystemExit("[02] notes_02.md contains a literal em-dash (CLAUDE.md hard rule 1)")
    NOTES_MD.write_text(text)



# ===========================================================================
# 7. main
# ===========================================================================
def prespecification() -> dict:
    """The LOCKED pre-specification (CLAUDE.md section 6, .build_facts.md section 2).

    Written into the fragment so `verify_numbers.py` can assert that nothing moved after the
    fact. Every value that exists as a contract constant is IMPORTED from `variable_dict`
    rather than retyped here, so the pre-specification the code enforces and the
    pre-specification the manuscript claims cannot drift apart.
    """
    out = {
        "prespec.locked_date": "2026-07-31",
        "prespec.time_zero": "the day of the principal operation, postoperative day 0",
        "prespec.estimator": ("Aalen-Johansen cumulative incidence function with death as "
                              "the competing event, not one minus Kaplan-Meier"),
        "prespec.stratum_comparison": "Gray's test on the cumulative incidence functions",
        "prespec.censoring": ("administrative at day 30, non-informative, complete by "
                              "construction"),
        "prespec.extrapolation": "none past postoperative day 30",
        "prespec.followup_days": int(VD.FOLLOWUP_DAYS),
        "prespec.tie_convention": VD.TIE_CONVENTION,
        "prespec.tie_convention_sensitivity": VD.TIE_CONVENTION_SENSITIVITY,
        # Both conventions carry a FORMULA key, not only prose, so verify_numbers.py can
        # assert the sensitivity convention mechanically instead of reading a sentence.
        "prespec.tie_convention_rule": "POSTDISCH = (POD > LOS_POSTOP)",
        "prespec.tie_convention_sensitivity_rule": "POSTDISCH = (POD >= LOS_POSTOP)",
        "prespec.window_composite": "WINDOW_COMP = " + " OR ".join(VD.WINDOW_COMPONENTS),
        "prespec.window_components": list(VD.WINDOW_COMPONENTS),
        "prespec.window_excluded": ["PNEUMONIA", "READMIT_UNPLANNED"],
        "prespec.composite_timing": ("the earliest observed postoperative day among the "
                                     "components that fired"),
        "prespec.risk_thresholds": [float(t) for t in VD.RISK_THRESHOLDS],
        "prespec.risk_threshold_labels": [VD.threshold_label(t) for t in VD.RISK_THRESHOLDS],
        "prespec.stratifiers": list(VD.STRATIFIER_COLUMNS),
        "prespec.min_stratum_events": int(VD.MIN_STRATUM_EVENTS),
        "prespec.suppression_rule": (
            f"a stratum with fewer than {VD.MIN_STRATUM_EVENTS} composite events is "
            f"suppressed rather than plotted, and the suppression is reported with its "
            f"count"),
        "prespec.seed": int(VD.SEED),
        "prespec.bootstrap_replicates": 1000,
        # .build_facts.md section 2's bootstrap row also locks WHAT is resampled. Without
        # this key that half of the row has nothing to assert against.
        "prespec.bootstrap_resampling_unit": "patients resampled within stratum",
        "prespec.bothways_covariates": list(VD.BOTHWAYS_COVARIATES),
        "prespec.steroid_excluded": True,
        "prespec.inout_denominator": "analytic",
        "prespec.inout_denominator_n": int(CPT.COHORT_LADDER_N["analytic"]),
        "prespec.tracheostomy_slot": (
            "reoperation slot 1 only inside the composite, the parent's frozen "
            "implementation; the slots 1 and 2 marker is ascertainment supplement only"),
        "prespec.los_field_primary": "DOPTODIS",
        "prespec.los_field_crosscheck": "TOTHLOS minus HTOODAY",
        "prespec.mortality_primary": "DOPERTOD",
        "prespec.mortality_sensitivity": "DOPERTOD or YRDEATH, never called 30-day mortality",
        "prespec.hematoma_definition": CPT.HEMATOMA_DEFINITION,
        "prespec.timing_range_min": TIMING_DAY_MIN,
        "prespec.timing_range_max": TIMING_DAY_MAX,
        "prespec.timing_out_of_range_rule": TIMING_RANGE_RULE,
        "prespec.timed_outcomes": list(VD.TIMED_OUTCOME_KEYS),
        "prespec.postdisch_ineligible_outcomes": [
            o.key for o in VD.TIMED_OUTCOMES if not o.postdisch_eligible],
        "prespec.no_hematoma_risk_model": True,
        "prespec.no_points_score": True,
    }
    for s in VD.STRATIFIERS:
        out[f"prespec.stratifier_levels.{s.column.lower()}"] = list(s.levels)
    return out


def build_results(d, analytic, counts_b, counts_a, masks_b, masks_a, g1) -> dict:
    """Assemble the whole fragment. Every key lands in one of this step's nine namespaces.

    `g1` is the Gate 1 block `main` already computed and printed. It is passed in rather
    than recomputed so the verdict the run printed and the verdict the fragment carries are
    the same object, and so the rebuild leg re-reads the bundle once per run, not twice.
    """
    a = analytic
    hem_a = _positive(a["HEMATOMA"])
    results: dict = {}

    # --- the two ladders --------------------------------------------------
    for step in LADDER_ORDER:
        n, ev = counts_b[step]
        results[f"cohort.{step}.n"] = n
        results[f"cohort.{step}.events"] = ev
        results[f"cohort.{step}.rate_pct"] = pct(ev, n)
        n2, ev2 = counts_a[step]
        results[f"gate.a.{step}.n"] = n2
        results[f"gate.a.{step}.events"] = ev2
        results[f"gate.a.{step}.rate_pct"] = pct(ev2, n2)
        results[f"cohort.nonagenarian_delta.{step}"] = n - n2
    # COMPUTED, never a literal: the verdict is the same comparison `assert_ladder` makes,
    # against the imported cpt_acdf constants, so the gate table in notes_02.md renders a
    # measurement and can print FAIL. The hard assertion remains the stop condition.
    results["gate.b.pass"] = bool(ladder_verdict(
        counts_b, CPT.COHORT_LADDER_N, CPT.COHORT_LADDER_EVENTS))
    results["gate.a.pass"] = bool(ladder_verdict(
        counts_a, CPT.COHORT_LADDER_N_PLAN, CPT.COHORT_LADDER_EVENTS))
    results["gate.b.rungs_checked"] = len(LADDER_ORDER) * 2
    results["gate.a.rungs_checked"] = len(LADDER_ORDER) * 2
    results["gate.b.note"] = (
        "Every rung and every event count asserted against cpt_acdf.COHORT_LADDER_N and "
        "COHORT_LADDER_EVENTS. No count is retyped in this step, and the verdict key is "
        "computed from that comparison rather than written as a constant.")
    results["gate.a.note"] = (
        "Reconciliation only. The delta is exactly the top-coded-age patients the plan's "
        "strict integer cast nulls, and it carries zero hematoma events. Its output belongs "
        "to the ascertainment supplement and never to the manuscript body.")
    results["cohort.hematoma.definition"] = CPT.HEMATOMA_DEFINITION
    results["cohort.hematoma.definition_label"] = \
        CPT.HEMATOMA_ICD10_VARIANTS[CPT.HEMATOMA_DEFINITION]["label"]

    # --- the three pre-specified outcome code sets, per rung --------------
    for variant in CPT.HEMATOMA_DEFINITION_ORDER:
        col = "HEMATOMA" if variant == CPT.HEMATOMA_DEFINITION else f"HEMATOMA_{variant}"
        flags = _positive(d[col])
        v = variant.lower()
        results[f"ladder.variant.{v}.label"] = CPT.HEMATOMA_ICD10_VARIANTS[variant]["label"]
        results[f"ladder.variant.{v}.is_primary"] = variant == CPT.HEMATOMA_DEFINITION
        for step in LADDER_ORDER:
            ev = int((masks_b[step] & flags).sum())
            results[f"ladder.variant.{v}.{step}.events"] = ev
            if ev != CPT.COHORT_LADDER_EVENTS_BY_DEFINITION[variant][step]:
                raise SystemExit(
                    f"[02] outcome variant {variant} at step {step!r} measures {ev}, "
                    f"pinned at {CPT.COHORT_LADDER_EVENTS_BY_DEFINITION[variant][step]}. "
                    f"CLAUDE.md 8.5: a mismatch is a stop condition, not a number to "
                    f"overwrite."
                )
            if int((masks_a[step] & flags).sum()) != ev:
                raise SystemExit(
                    f"[02] outcome variant {variant} at step {step!r} differs between "
                    f"Gate A and Gate B. The restored top-coded-age patients carry zero "
                    f"hematoma events under every definition; that no longer holds."
                )
    results["cohort.analytic.events_v0"] = results["ladder.variant.v0.analytic.events"]
    results["cohort.analytic.events_v1"] = results["ladder.variant.v1.analytic.events"]

    # --- the hematoma slot convention, primary definition, analytic rung --
    _s2_cpt = _flag_col(d[REOP_SLOT2_CPT_COLUMN], CPT.is_hematoma_cpt).to_numpy(dtype=bool)
    _s2_n9 = d["REOPOR2ICD91"].map(CPT.normalize_icd)
    _s2_n10 = d["REOPOR2ICD101"].map(CPT.normalize_icd)
    _spec = CPT.HEMATOMA_ICD10_VARIANTS[CPT.HEMATOMA_DEFINITION]
    _s2_ok10 = ((_s2_n10.isin(_spec["exact"]) | _s2_n10.str.startswith(_spec["prefixes"]))
                & (_s2_n10 != "") & ~_s2_n10.isin(_spec["seroma"]))
    _s2 = pd.Series(_s2_cpt, index=d.index) | _s2_n9.isin(CPT.HEMATOMA_ICD9) | _s2_ok10
    _s1 = _positive(d["HEMATOMA"])
    m = masks_b["analytic"]
    results["derived.slot_union.convention"] = "Reoperation slot 1 only"
    results["derived.slot_union.analytic.slot1_events"] = int((m & _s1).sum())
    results["derived.slot_union.analytic.slots_1_2_events"] = int((m & (_s1 | _s2)).sum())
    results["derived.slot_union.analytic.slot2_added_events"] = int((m & ~_s1 & _s2).sum())
    results["derived.slot_union.note"] = (
        "The primary outcome reads the first reoperation slot only, which is the convention "
        "the reconciliation gates are defined on. Reading both slots would add the cases "
        "whose hematoma reoperation was recorded as the patient's second return to theatre.")

    # --- STROBE boxes and the exclusion arithmetic ------------------------
    results["cohort.flow.not_elective.n"] = int(
        (masks_b["identified"] & ~masks_b["elective"]).sum())
    for name, col in (("excl_corpectomy", "EXCL_CORPECTOMY"),
                      ("excl_posterior", "EXCL_POSTERIOR"),
                      ("excl_arthroplasty", "EXCL_ARTHROPLASTY")):
        results[f"cohort.flow.{name}.n"] = int(
            (masks_b["elective"] & _positive(d[col])).sum())
    results["cohort.flow.excl_procedure_any.n"] = int(
        (masks_b["elective"] & ~masks_b["exclusions"]).sum())
    final_boxes = 0
    for name, col in (("excl_asa5", "EXCL_ASA5"),
                      ("excl_asa_unassigned", "EXCL_ASA_UNASSIGNED"),
                      ("excl_presepsis", "EXCL_PRESEPSIS")):
        k = int((masks_b["exclusions"] & _positive(d[col])).sum())
        results[f"cohort.flow.{name}.n"] = k
        final_boxes += k
    # CLAUDE.md 9.16: the rung difference and the box sum differ by exactly the number of
    # cases meeting more than one criterion. STATE the arithmetic; do NOT adjust a count.
    delta = CPT.COHORT_LADDER_N["exclusions"] - CPT.COHORT_LADDER_N["analytic"]
    results["cohort.flow.exclusion_delta"] = int(delta)
    results["cohort.flow.exclusion_boxes_sum"] = int(final_boxes)
    results["cohort.flow.exclusion_overlap_n"] = int(final_boxes - delta)
    results["cohort.flow.exclusion_boxes_disjoint"] = bool(final_boxes == delta)
    results["cohort.flow.exclusion_boxes_sum_recorded"] = CONSTITUTION_EXCLUSION_BOX_SUM
    results["cohort.flow.exclusion_arithmetic_reproduces"] = bool(
        final_boxes == CONSTITUTION_EXCLUSION_BOX_SUM)
    results["cohort.flow.exclusion_arithmetic_note"] = (
        f"The final rung removes {delta} cases and the three exclusion boxes sum to "
        f"{final_boxes}, so {final_boxes - delta} case meets more than one exclusion "
        f"criterion. State the arithmetic; do not adjust a count to make it close.")

    # --- ported completeness flags (never fitted here) --------------------
    mc, lab = _positive(a["MODEL_COMPLETE"]), _positive(a["LAB_COMPLETE"])
    results["derived.subgroup.model_complete.n"] = int(mc.sum())
    results["derived.subgroup.model_complete.events"] = int((mc & hem_a).sum())
    results["derived.subgroup.model_complete.pct"] = pct(int(mc.sum()), len(a))
    results["derived.subgroup.inr_measured.n"] = int(a["INR_MEASURED"].sum())
    results["derived.subgroup.lab_complete.n"] = int(lab.sum())
    results["derived.subgroup.note"] = (
        "Ported completeness flags. This project fits no hematoma risk model and no burden "
        "analysis; the flags exist so the two cohorts carry the same columns.")

    # --- BMI window, sex, levels, operative time, eras --------------------
    results["derived.bmi.window_min"] = BMI_MIN
    results["derived.bmi.window_max"] = BMI_MAX
    results["derived.bmi.implausible_analytic_n"] = int(a["_BMI_IMPLAUSIBLE"].sum()) \
        if "_BMI_IMPLAUSIBLE" in a.columns else int(
            (masks_b["analytic"] & d["_BMI_IMPLAUSIBLE"]).sum())
    results["derived.bmi.implausible_analytic_events"] = int(
        (masks_b["analytic"] & d["_BMI_IMPLAUSIBLE"] & _positive(d["HEMATOMA"])).sum())
    results["derived.bmi.missing_analytic_n"] = int(a["BMI"].isna().sum())
    results["derived.sex.nonbinary_analytic_n"] = int((a["SEX"] == "non-binary").sum())
    results["derived.levels.cap"] = LEVELS_CAP
    results["derived.levels.max"] = int(a["LEVELS_N"].max())
    results["derived.levels.above_cap_n"] = int((a["LEVELS_N"] > LEVELS_CAP).sum())
    results["derived.optime.missing_n"] = int(a["OPTIME"].isna().sum())
    results["derived.era.icd10_n"] = int((a["ERA_ICD10"] == 1).sum())
    results["derived.era.emergncy_n"] = int((a["ERA_EMERGNCY"] == 1).sum())
    results["derived.era.icd10_first_year"] = 2016
    results["derived.era.emergncy_last_year"] = 2020

    # --- per-year cohort and events, and the ICD-9-only ascertainment -----
    rates = []
    for year, k in sorted(a["PUFYEAR"].value_counts().items()):
        ev = int(hem_a[a["PUFYEAR"] == year].sum())
        results[f"cohort.analytic.by_year.{int(year)}.n"] = int(k)
        results[f"cohort.analytic.by_year.{int(year)}.events"] = ev
        rates.append(100.0 * ev / int(k))
    results["derived.rate_by_year.min_pct"] = round(min(rates), 4)
    results["derived.rate_by_year.max_pct"] = round(max(rates), 4)
    ident = d[masks_b["identified"]]
    only9 = (_flag_col(ident[CPT.HEMATOMA_ICD9_COLUMN], CPT.is_hematoma_icd9)
             & ~_flag_col(ident[CPT.HEMATOMA_CPT_COLUMN], CPT.is_hematoma_cpt)
             & ~_flag_col(ident[CPT.HEMATOMA_ICD10_COLUMN], CPT.is_hematoma_icd10)
             ).astype(bool)
    results["derived.icd9_only.events_identified"] = int(only9.sum())
    results["derived.icd9_only.last_year"] = int(ident.loc[only9, "PUFYEAR"].max())
    results["derived.icd9_only.events_2016"] = int((only9 & (ident["PUFYEAR"] == 2016)).sum())

    # --- the measurement blocks ------------------------------------------
    results.update(g1)
    results.update(preflight_timing(a))
    results.update(negative_flag_days(a))
    results.update(second_events(a))
    results.update(carryover_reconciliation(d, masks_b))
    results.update(trach_invariant(d, masks_b["analytic"]))
    results.update(expired_reconciliation(a))
    results.update(cross_checks(d, masks_b["analytic"]))
    results.update(prespecification())

    # --- schema bookkeeping ----------------------------------------------
    results["derived.schema.n_columns"] = int(a.shape[1])
    results["derived.schema.n_documented"] = len(SCHEMA_SPEC)
    results["derived.schema.labels_added"] = len(LABELS_ADDED)
    results["derived.schema.labels_added_keys"] = list(LABELS_ADDED)
    results["derived.schema.columns_without_labels"] = len(sorted(
        c for c in a.columns
        if c not in VD.LABELS and not c.startswith(("OTHERCPT", "CONCPT"))))
    return results


def clean_fragment(results: dict) -> dict:
    """Validate and normalise the fragment before it is written.

    `run_all.py` enforces the namespaces at merge time and rejects the whole run on the first
    violation. Doing it here as well means a violation fails inside the step that caused it,
    with the offending key named, rather than three steps later.
    """
    out = {}
    bad_ns, bad_key, bad_val = [], [], []
    for key, value in results.items():
        if not KEY_RE.match(key):
            bad_key.append(key)
        if not key.startswith(RESERVED_PREFIXES) and key not in tuple(
                p.rstrip(".") for p in RESERVED_PREFIXES):
            bad_ns.append(key)
        v = value
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        if isinstance(v, float) and not math.isfinite(v):
            v = None                       # JSON has no NaN; write null (notes_run_all.md 3)
        elif isinstance(v, (list, tuple)):
            items = []
            for item in v:
                if isinstance(item, (np.integer,)):
                    item = int(item)
                elif isinstance(item, (np.floating,)):
                    item = float(item)
                elif isinstance(item, (np.bool_,)):
                    item = bool(item)
                if isinstance(item, float) and not math.isfinite(item):
                    item = None
                if not isinstance(item, (str, int, float, bool)) and item is not None:
                    bad_val.append(key)
                items.append(item)
            v = items
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
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
        raise SystemExit("[02] fragment contract violation:\n       "
                         + "\n       ".join(problems))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="02_build_cohort.py",
        description="Recode the export bundle and freeze the analytic cohort.",
    )
    ap.add_argument("--replicate-plan", action="store_true",
                    help="Gate A reconciliation: reproduce the plan's strict AGE cast. "
                         "Asserts, prints and WRITES NOTHING.")
    args = ap.parse_args(argv)

    mode = "GATE A (--replicate-plan, reconciliation only)" if args.replicate_plan \
        else "GATE B (default, PRIMARY)"
    print("=" * 78)
    print("[02] ACDF discharge timing -- cohort builder")
    print(f"[02] mode  : {mode}")
    print(f"[02] seed  : {VD.SEED}")
    print(f"[02] input : {EXPORT_CSV}")
    print("=" * 78)

    d = read_bundle()
    n_columns = d.shape[1]
    print(f"[02]   bundle {len(d):,} rows x {n_columns} columns "
          f"(keep_default_na=False; PRSEPIS 'None' preserved)")
    d = derive(d)

    # --- both ladders, always --------------------------------------------
    masks_b = ladder_masks(d, replicate_plan=False)
    masks_a = ladder_masks(d, replicate_plan=True)
    counts_b = ladder_counts(d, masks_b)
    counts_a = ladder_counts(d, masks_a)

    for step in LADDER_ORDER:
        delta = counts_b[step][0] - counts_a[step][0]
        want = CPT.COHORT_LADDER_NONAGENARIAN_DELTA[step]
        if delta != want:
            raise SystemExit(
                f"[02] the strict-cast delta at '{step}' is {delta}, expected {want}. "
                f"The top-coded-age reconstruction is wrong; stop."
            )

    print_ladder(counts_b, CPT.COHORT_LADDER_N, CPT.COHORT_LADDER_EVENTS,
                 "Gate B ladder (corrected age handling, PRIMARY)")
    assert_ladder(counts_b, CPT.COHORT_LADDER_N, CPT.COHORT_LADDER_EVENTS, "Gate B")
    print("[02]   Gate B PASS -- all 8 counts exact.")
    print("")
    print_ladder(counts_a, CPT.COHORT_LADDER_N_PLAN, CPT.COHORT_LADDER_EVENTS,
                 "Gate A ladder (plan's strict integer age cast, reconciliation)")
    assert_ladder(counts_a, CPT.COHORT_LADDER_N_PLAN, CPT.COHORT_LADDER_EVENTS, "Gate A")
    print("[02]   Gate A PASS -- all 8 counts exact.")
    print("")

    g1 = gate_one(d, counts_b, counts_a, n_columns)
    print_gate_one(g1)
    print("")

    if args.replicate_plan:
        a = d[masks_a["analytic"]]
        print("[02] Gate A analytic reference counts (reconciliation):")
        print(f"[02]   WINDOW_COMP {int(a['WINDOW_COMP'].sum()):,}   "
              f"AIRWAY_COMP {int(a['AIRWAY_COMP'].sum()):,}   "
              f"MODEL_COMPLETE {int(a['MODEL_COMPLETE'].sum()):,}")
        print("[02] --replicate-plan writes NOTHING by design; the frozen artefacts belong "
              "to the default mode.")
        print("=" * 78)
        print("[02] DONE. Gate A reconciliation PASSED.")
        print("=" * 78)
        return 0

    # ================= default mode: freeze the artefacts =================
    if not g1["gate.1.pass"]:
        raise SystemExit(
            "[02] GATE 1 FAILED. CLAUDE.md 8.3 makes this a stop condition. Nothing was "
            "written."
        )

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    d["STEP_ELECTIVE"] = _i8(masks_b["elective"].astype(float), d.index)
    d["STEP_EXCLUSIONS"] = _i8(masks_b["exclusions"].astype(float), d.index)
    d["STEP_ANALYTIC"] = _i8(masks_b["analytic"].astype(float), d.index)

    internal = [c for c in d.columns if c.startswith("_")]
    identified = d[masks_b["identified"]].drop(columns=internal).reset_index(drop=True)
    analytic = d[masks_b["analytic"]].drop(columns=internal).reset_index(drop=True)

    results = build_results(d, analytic, counts_b, counts_a, masks_b, masks_a, g1)

    # The timing-rule delta on the hematoma day column, measured rather than assumed: the
    # parent's column is the raw days field masked by the outcome, this one additionally
    # applies the handling rule, and the two must be reported if they ever differ.
    _ha = d.loc[masks_b["analytic"]]
    raw_pod = int(_ha["_HEMATOMA_POD_RAW"].notna().sum())
    use_pod = int(_ha["HEMATOMA_POD"].notna().sum())
    results["dq.hematoma_pod.raw_present_n"] = raw_pod
    results["dq.hematoma_pod.usable_present_n"] = use_pod
    results["dq.hematoma_pod.dropped_by_range_rule_n"] = raw_pod - use_pod
    results["dq.hematoma_pod.matches_parent_form"] = bool(raw_pod == use_pod)

    # .build_facts.md 9.6: declare the unit of every family this step publishes, LAST, so
    # the coverage assertion sees the complete fragment. Additive metadata only; nothing
    # above is read, renamed or recomputed here.
    n_families = assert_scale_coverage(results)
    publish_scale_declarations(results)
    print(f"[02] scale declarations: {len(SCALE_DECLARATIONS)} namespaces, {n_families} "
          f"families, each scoped to its own namespace (.build_facts.md 9.6)")
    print("")

    print("[02] pre-flight data check (CLAUDE.md 8.6)")
    print(f"[02]   days-to-event fields audited: {len(TIMING_FIELDS)}; "
          f"values outside 0-30 or non-integer: "
          f"{results['timingfield.total_dropped_by_range_rule_n']}; "
          f"all clean integer days: {results['timingfield.all_fields_clean_integer_days']}")
    print(f"[02]   {'outcome':<10} {'events':>7} {'day obs':>8} {'% missing':>10} "
          f"{'both':>7} {'lost 1-4':>9}")
    for k in VD.TIMED_OUTCOME_KEYS:
        print(f"[02]   {k:<10} {results[f'timingfield.{k}.events']:>7,} "
              f"{results[f'timingfield.{k}.pod_present_n']:>8,} "
              f"{results[f'timingfield.{k}.pod_missing_pct']:>9.2f}% "
              f"{results[f'timingfield.{k}.both_present_n']:>7,} "
              f"{results[f'timingfield.{k}.lost_rows_1_to_4_pct']:>8.2f}%")
    print(f"[02]   WHY the events between rows 1 and 4 are lost (nothing is reclassified):")
    print(f"[02]   {'outcome':<10} {'lost':>5} {'day miss':>9} {'LOS miss, still':>16} "
          f"{'LOS miss, not still':>20} {'bound naive':>12} {'bound cond':>11}")
    for k in VD.TIMED_OUTCOME_KEYS:
        print(f"[02]   {k:<10} "
              f"{results[f'timingfield.{k}.lost_rows_1_to_4_n']:>5,} "
              f"{results[f'timingfield.{k}.lost_day_missing_n']:>9,} "
              f"{results[f'timingfield.{k}.lost_los_missing_stillinhosp_n']:>16,} "
              f"{results[f'timingfield.{k}.lost_los_missing_not_stillinhosp_n']:>20,} "
              f"{results[f'timingfield.{k}.bound_unconditional_max_n']:>12,} "
              f"{results[f'timingfield.{k}.bound_stillinhosp_conditioned_max_n']:>11,}")
    print(f"[02]   every length-of-stay loss belongs to a still-in-hospital patient: "
          f"{results['timingfield.loss_all_los_loss_is_stillinhosp']}. "
          f"04 reports the STILLINHOSP-conditioned bound as PRIMARY.")
    print(f"[02]   still in hospital beyond 30 days {results['losfield.stillinhosp_yes_n']}: "
          f"{results['losfield.stillinhosp_los_missing_n']} with no length of stay, "
          f"{results['losfield.stillinhosp_los_present_n']} with one "
          f"({results['losfield.stillinhosp_los_le30_n']} of 30 days or fewer); "
          f"{results['losfield.stillinhosp_postdisch_observed_n']} are already inside "
          f"04's row-4 denominators")
    print(f"[02]   same-day discharge {results['losfield.sameday_disch_n']:,} contains "
          f"{results['losfield.sameday_disch_expired_dest_n']} expired, "
          f"{results['losfield.sameday_disch_mort30_n']} recorded deaths and "
          f"{results['losfield.sameday_disch_stillinhosp_n']} still in hospital; "
          f"discharge by POD 1 {results['losfield.disch_by_pod1_n']:,} contains "
          f"{results['losfield.disch_by_pod1_expired_dest_n']}, "
          f"{results['losfield.disch_by_pod1_mort30_n']} and "
          f"{results['losfield.disch_by_pod1_stillinhosp_n']}. 05 must exclude them.")
    print(f"[02]   length of stay: DOPTODIS vs TOTHLOS minus HTOODAY, both present "
          f"{results['losfield.both_present']:,}, disagree "
          f"{results['losfield.disagree_n']:,} "
          f"(parent recorded {results['losfield.parent_disagree_n']} on "
          f"{results['losfield.parent_both_present']:,}: "
          f"{'REPRODUCED' if results['losfield.reproduces_parent'] else 'NOT REPRODUCED'})")
    print(f"[02]   expired destination {results['dq.expired.n']}, of which "
          f"{results['dq.expired.no_mort30_n']} carry no death day; "
          f"{results['dq.mort30.dischdest_not_expired_n']} recorded deaths have a "
          f"non-expired destination (died after discharge)")
    if results["derived.trach.invariant_holds"]:
        print(f"[02]   tracheostomy invariant HOLDS on the analytic cohort: the airway "
              f"composite is identical on slot 1 "
              f"({results['derived.trach.slot1_analytic_n']}) and slots 1+2 "
              f"({results['derived.trach.slot12_analytic_n']}), "
              f"{results['derived.trach.invariant_rows_differing_analytic']} rows differ; "
              f"all {results['derived.trach.slot2_only_analytic_n']} slot-2-only cases "
              f"already enter through reintubation or prolonged ventilation")
        if not results["derived.trach.invariant_holds_identified"]:
            print(f"[02]   NOTE, published not suppressed: on the wider IDENTIFIED cohort "
                  f"the two readings differ on "
                  f"{results['derived.trach.invariant_rows_differing_identified']} "
                  f"patient(s), every one excluded by the cohort definition before any "
                  f"analysis sees them. The composite is only ever computed on the "
                  f"analytic cohort.")
    else:
        print("[02]   BLOCKER: THE TRACHEOSTOMY INVARIANT NO LONGER HOLDS ON THE ANALYTIC "
              "COHORT. CLAUDE.md 6.2 requires the airway composite to be identical whether "
              "the tracheostomy arm reads slot 1 or slots 1+2, and it is not. Stop and "
              "re-rule before any curve is fitted. The composite has NOT been changed; "
              f"{results['derived.trach.invariant_rows_differing_analytic']} analytic rows "
              "differ.")
    if not results["cohort.flow.exclusion_arithmetic_reproduces"]:
        print(f"[02]   NOTE: the exclusion boxes sum to "
              f"{results['cohort.flow.exclusion_boxes_sum']}, where CLAUDE.md 9.16 records "
              f"{CONSTITUTION_EXCLUSION_BOX_SUM}. Reported, not adjusted (CLAUDE.md 8.5).")
    print("")

    analytic.to_parquet(ANALYTIC_PARQUET, index=False)
    identified.to_parquet(IDENTIFIED_PARQUET, index=False)
    print(f"[02]   wrote {ANALYTIC_PARQUET}   {len(analytic):,} x {analytic.shape[1]}")
    print(f"[02]   wrote {IDENTIFIED_PARQUET} {len(identified):,} x {identified.shape[1]}")

    write_schema_md(analytic, results)
    print(f"[02]   wrote {SCHEMA_MD}")

    fragment = clean_fragment(results)
    RESULTS_JSON.write_text(json.dumps(fragment, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")
    print(f"[02]   wrote {RESULTS_JSON}  ({len(fragment)} keys, namespaces validated)")

    write_notes_md(results)
    print(f"[02]   wrote {NOTES_MD}")

    # --- read-back verification ------------------------------------------
    back = pd.read_parquet(ANALYTIC_PARQUET)
    if len(back) != CPT.COHORT_LADDER_N["analytic"]:
        raise SystemExit(f"[02] parquet read-back has {len(back):,} rows")
    if int(back["HEMATOMA"].sum()) != CPT.COHORT_LADDER_EVENTS["analytic"]:
        raise SystemExit("[02] parquet read-back event count is wrong")
    missing_spec = [c for c, *_ in SCHEMA_SPEC if c not in back.columns]
    if missing_spec:
        raise SystemExit(f"[02] parquet read-back is missing {missing_spec}")
    contract = (list(VD.POD_COLUMNS) + list(VD.POSTDISCH_COLUMNS)
                + list(VD.TIMED_BOTH_COLUMNS)
                + ["INOUT", "INOUT_OUTPT", "LOS_POSTOP", "LOS_POSTOP_CHECK",
                   "LOS_POSTOP_AGREE", "LOS_CAT", "SAMEDAY_DISCH", "DISCH_BY_POD1",
                   "AGE_CAT", "REOP_TRACH_SLOT12", "WINDOW_COMP", "POD_TRACH",
                   "POD_TRACH12", "POD_WINDOW", "ERA_ICD10", "ERA_EMERGNCY"])
    missing_contract = [c for c in contract if c not in back.columns]
    if missing_contract:
        raise SystemExit(
            f"[02] the frozen parquet is missing contracted columns {missing_contract} "
            f"(.build_facts.md section 5)."
        )
    if "POD_HEMATOMA" in back.columns:
        raise SystemExit("[02] POD_HEMATOMA was created; the roster uses HEMATOMA_POD")
    print(f"[02]   read-back OK: {len(back):,} rows x {back.shape[1]} columns, "
          f"{int(back['HEMATOMA'].sum())} events, all {len(SCHEMA_SPEC)} documented and "
          f"{len(contract)} contracted columns present.")

    print("")
    print("=" * 78)
    print("[02] SUMMARY")
    print(f"[02]   analytic   n = {results['cohort.analytic.n']:,}   "
          f"events = {results['cohort.analytic.events']}   "
          f"rate = {results['cohort.analytic.rate_pct']:.3f}%   "
          f"definition = {results['cohort.hematoma.definition']}")
    print(f"[02]   identified n = {results['cohort.identified.n']:,}")
    print(f"[02]   Gate B PASS   Gate A PASS   "
          f"Gate 1 {'PASS' if results['gate.1.pass'] else 'FAIL'}")
    print(f"[02]   WINDOW_COMP {results['derived.window_comp.n']} "
          f"({results['derived.window_comp.rate_pct']:.3f}%)   "
          f"AIRWAY_COMP {results['derived.airway.composite_n']}   "
          f"same-day discharge {results['losfield.sameday_disch_n']:,}")
    print("[02] DONE.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
