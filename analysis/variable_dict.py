"""Display and label layer for the ACDF discharge-timing study (ACS-NSQIP 2012-2024).

This module is the ONLY sanctioned source of user-visible text for variable names and
category levels in this project, the only definition of the project seed, and the SINGLE
SOURCE OF TRUTH for the `DISCHDEST` and `INOUT` level maps. It is ported from the
companion hematoma project's `variable_dict.py`; `VarSpec`,
`short` / `prose` / `unit` / `var_type`, `pretty`, `pretty_level`, `has_snake_case`,
`has_em_dash`, `EN_DASH` and the `fmt_*` family are carried over with their house
behaviour. `LABELS`, `SECTIONS`, the contract constants and every `*_PRETTY` dict are this
study's own content.

Four things every downstream script must honor:

1. **No snake_case token may reach rendered text.** `pretty()` (variable names) and
   `pretty_level()` (category levels) are the ONLY entry points for user-visible text.
   `short()` / `prose()` / `unit()` / `var_type()` are raw accessors on `LABELS`;
   `pretty()` delegates to `short()` for every key in `LABELS`, so the two can never
   disagree. `has_snake_case()` is the assertion helper and matches anywhere in a string.
2. **No EM-dash (U+2014) in rendered text.** `has_em_dash()` is the assertion helper.
   EN_DASH (U+2013) is the range separator used by `fmt_iqr` / `fmt_or_ci` / `fmt_rr_ci`
   and by `AGE_CAT_PRETTY`, and is CORRECT. Do not strip it. `assert_renderable()` checks
   both bans at once and is what `07_figures.py` and `08_generate_manuscript.py` should
   call at write time.
3. **A bare `1.0` is also a defect.** `has_snake_case()` cannot catch it, so `pretty_level()`
   catches it instead: every boolean-ish spelling NSQIP uses (`1`, `1.0`, `True`, `"Yes"`,
   `"YES"`, `"yes"`) folds onto one affirmative label, the false side onto one negative
   label, and NaN / None onto `MISSING_LABEL`. NSQIP value casing varies by year
   (`DIABETES` is `NO` while `BLEEDDIS` is `No`), so level lookup is case-insensitive.
4. **`SEED` lives here and nowhere else.** Import it; never re-declare it.

Run `python3 variable_dict.py` to execute the self-test at the bottom of this file. It
exits 0 on success and non-zero on any failure.

Naming convention this file pins down, and which `02_build_cohort.py` MUST follow:

  * RAW warehouse columns keep their NSQIP spelling exactly (`AGE`, `INOUT`, `DOPTODIS`,
    `DREINTUB`, `RETORPODAYS`). Do not rename them in the analytic parquet.
  * DERIVED analytic columns are UPPER_SNAKE and descriptive (`LOS_CAT`, `AGE_CAT`,
    `POD_REINTUB`, `POSTDISCH_HEMATOMA`, `WINDOW_COMP`). Every derived column named in
    `LABELS` below is a name the builder is expected to produce verbatim; downstream steps
    look it up by that exact key.
  * Categorical LEVEL strings stored in the parquet are ASCII and stable (`">=3"`, not
    a Unicode glyph). The display form lives on the right-hand side of a `*_PRETTY` dict,
    never in the data.

What is NEW here relative to the parent, and why:

  * `DISCHDEST_LEVEL_MAP` and `INOUT_LEVEL_MAP` move HERE and become the single source of
    truth. `02_build_cohort.py` imports them instead of carrying its own copy. Both raise
    on an unmapped level rather than defaulting, because a future PUF vintage quietly
    folding a new level into "Home" is precisely the failure the map exists to prevent.
  * `TIMED_OUTCOMES` is the eight-entry roster every step from `02` to `07` iterates, in
    one order, with the event-flag column, the postoperative-day column and the raw NSQIP
    timing field attached to each. The `POD_*`, `POSTDISCH_*` and `TIMED_BOTH_*` column
    names and their labels are GENERATED from it, so 24 names cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# PROJECT SEED
# ---------------------------------------------------------------------------
SEED: int = 20260731
"""The one project seed. Defined here, imported
everywhere else. Never re-declare it, never pass a literal. It is 20260731, this
project's own seed, and deliberately NOT the parent project's 20260730.

Callers must do BOTH of the following, because the two cover different random streams:

    from variable_dict import SEED
    import numpy as np

    np.random.seed(SEED)                       # the global NumPy stream
    estimator.fit(X, y, random_state=SEED)     # any API that accepts an explicit seed

`np.random.seed(SEED)` alone does not seed a library estimator that defaults to
`random_state=None`, and `random_state=SEED` alone does not seed a bare
`np.random.choice` inside a hand-rolled bootstrap.

In THIS project exactly one step draws on the global NumPy stream: the 1,000-replicate
conditional-risk bootstrap in `06_window.py`, which resamples patients within stratum.
Every other step is deterministic; they still call `np.random.seed(SEED)`
at the top so that adding a draw later cannot silently unseed the pipeline.
"""


class VarSpec(NamedTuple):
    short_label: str
    prose_label: str
    units: Optional[str]
    var_type: str  # "continuous" | "binary" | "categorical" | "ordinal"


# ---------------------------------------------------------------------------
# THE TIMED-OUTCOME ROSTER
#
# This lives ABOVE `LABELS` because the `POD_*`, `POSTDISCH_*` and `TIMED_BOTH_*` label
# entries are generated from it immediately after the `LABELS` literal. It is repeated in
# the contract-constants section header so a reader looking for the constants finds it.
#
# ORDER is the order in which Figure 2 lists the panels: hematoma reoperation, any
# unplanned reoperation, unplanned reintubation, ventilator beyond 48 hours, pneumonia,
# unplanned readmission, death; then composite airway morbidity. `02` through `07` all
# iterate THIS tuple, in THIS order, so a table row and a figure panel cannot disagree.
# ---------------------------------------------------------------------------
class TimedOutcome(NamedTuple):
    """One timed outcome and every column name that hangs off it.

    Fields:
      key                 lowercase analysis key. It is the token used inside
                          `outputs/results_0N.json` dotted keys, e.g. "postdisch.hematoma.pct".
      suffix              UPPER token that names the derived column family:
                          `POD_<suffix>` (except hematoma, see `pod_col`),
                          `POSTDISCH_<suffix>`, `TIMED_BOTH_<suffix>`.
      noun                lowercase mid-sentence clinical noun. Every generated label is
                          built from it, so the three families cannot describe the same
                          outcome three different ways.
      event_col           the event-flag column in `derived/cohort_analytic.parquet`. It is
                          also the `LABELS` key that names the outcome.
      pod_col             the postoperative-day column. `POD_<suffix>` for every outcome
                          EXCEPT hematoma, which reuses the parent's `HEMATOMA_POD`
                          (do not create a second `POD_HEMATOMA` column).
      raw_fields          the raw NSQIP timing field(s) the day column is taken from. A
                          tuple because the airway composite is the earliest of three.
                          This is what supplement S2 tabulates missingness for.
      postdisch_eligible  whether the post-discharge FRACTION is an interpretable quantity
                          for this outcome. False for unplanned readmission only:
                          `READMPODAYS1` is days from the principal operation to
                          readmission, so a readmission is a post-discharge event BY
                          CONSTRUCTION for anyone who was discharged (the readmission
                          caveat). Its curve
                          SHAPE is reported; its post-discharge fraction is not.
    """
    key: str
    suffix: str
    noun: str
    event_col: str
    pod_col: str
    raw_fields: Tuple[str, ...]
    postdisch_eligible: bool


TIMED_OUTCOMES: Tuple[TimedOutcome, ...] = (
    TimedOutcome(
        key="hematoma", suffix="HEMATOMA", noun="hematoma reoperation",
        event_col="HEMATOMA", pod_col="HEMATOMA_POD",
        raw_fields=("RETORPODAYS",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="reop_any", suffix="REOP_ANY", noun="unplanned reoperation",
        event_col="REOP_ANY", pod_col="POD_REOP_ANY",
        raw_fields=("RETORPODAYS",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="reintub", suffix="REINTUB", noun="unplanned reintubation",
        event_col="REINTUBATION", pod_col="POD_REINTUB",
        raw_fields=("DREINTUB",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="vent48", suffix="VENT48", noun="ventilator dependence beyond 48 hours",
        event_col="VENT48", pod_col="POD_VENT48",
        raw_fields=("DFAILWEAN",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="pneumonia", suffix="PNEUMONIA", noun="pneumonia",
        event_col="PNEUMONIA", pod_col="POD_PNEUMONIA",
        raw_fields=("DOUPNEUMO",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="readmit", suffix="READMIT", noun="unplanned readmission",
        event_col="READMIT_UNPLANNED", pod_col="POD_READMIT",
        raw_fields=("READMPODAYS1",), postdisch_eligible=False,
    ),
    TimedOutcome(
        key="death", suffix="DEATH", noun="death",
        event_col="MORT30", pod_col="POD_DEATH",
        raw_fields=("DOPERTOD",), postdisch_eligible=True,
    ),
    TimedOutcome(
        key="airway", suffix="AIRWAY", noun="composite airway morbidity",
        event_col="AIRWAY_COMP", pod_col="POD_AIRWAY",
        raw_fields=("DREINTUB", "DFAILWEAN", "RETORPODAYS"), postdisch_eligible=True,
    ),
)

TIMED_OUTCOME_KEYS: Tuple[str, ...] = tuple(_o.key for _o in TIMED_OUTCOMES)
_TIMED_BY_KEY: Dict[str, TimedOutcome] = {_o.key: _o for _o in TIMED_OUTCOMES}

# The three generated column families, in roster order. `02` produces them; `03` to `07`
# read them. Nothing in this project may type one of these names by hand.
POD_COLUMNS: Tuple[str, ...] = tuple(_o.pod_col for _o in TIMED_OUTCOMES)
POSTDISCH_COLUMNS: Tuple[str, ...] = tuple(f"POSTDISCH_{_o.suffix}" for _o in TIMED_OUTCOMES)
TIMED_BOTH_COLUMNS: Tuple[str, ...] = tuple(f"TIMED_BOTH_{_o.suffix}" for _o in TIMED_OUTCOMES)


def timed_outcome(key: str) -> TimedOutcome:
    """The roster entry for an analysis key. Raises KeyError on an unknown key."""
    return _TIMED_BY_KEY[key]


def pod_column(key: str) -> str:
    """The postoperative-day column for an analysis key."""
    return _TIMED_BY_KEY[key].pod_col


def postdisch_column(key: str) -> str:
    """The primary-tie-convention post-discharge flag column for an analysis key."""
    return f"POSTDISCH_{_TIMED_BY_KEY[key].suffix}"


def timed_both_column(key: str) -> str:
    """Row 4 of the denominator ladder: both timing fields present."""
    return f"TIMED_BOTH_{_TIMED_BY_KEY[key].suffix}"


# ---------------------------------------------------------------------------
# LABELS: every variable this study touches in a table, figure, or prose string.
#
# Keys come in two flavors:
#   - RAW warehouse column names, NSQIP spelling preserved (AGE, INOUT, DOPTODIS, ...).
#   - DERIVED analytic columns produced by 02_build_cohort.py, UPPER_SNAKE and descriptive
#     (LOS_CAT, AGE_CAT, WINDOW_COMP, POD_AIRWAY, ...).
#
# Short labels are written as reader-facing table stubs: clinically readable prose with
# the unit attached where a unit exists. Prose labels are the lowercase mid-sentence form
# used inside manuscript sentences.
#
# The column schema written by `02_build_cohort.py` is the authoritative list of what that
# script produces. Every name there has an entry here; neither file may invent a name the other
# does not have.
# ---------------------------------------------------------------------------
LABELS: Dict[str, VarSpec] = {

    # ======================================================================
    # DEMOGRAPHICS AND ANTHROPOMETRICS
    # ======================================================================
    "AGE": VarSpec("Age, years", "age", "years", "continuous"),
    "AGE_PER10": VarSpec("Age, per decade", "age per additional decade", "decades", "continuous"),
    # NEW: the locked age stratifier. Levels are ASCII in the parquet.
    "AGE_CAT": VarSpec("Age group", "age group", None, "categorical"),
    "SEX": VarSpec("Sex", "sex as recorded", None, "categorical"),
    "MALE": VarSpec("Male sex", "male sex", None, "binary"),
    "RACE_NEW": VarSpec("Race", "race as recorded", None, "categorical"),
    "ETHNICITY_HISPANIC": VarSpec("Hispanic ethnicity", "Hispanic ethnicity", None, "categorical"),
    "RACE_CAT": VarSpec("Race and ethnicity", "race and ethnicity", None, "categorical"),
    "HEIGHT": VarSpec("Height, inches", "height", "inches", "continuous"),
    "WEIGHT": VarSpec("Weight, pounds", "weight", "pounds", "continuous"),
    "BMI": VarSpec("Body mass index, kg/m²", "body mass index", "kg/m²", "continuous"),
    "BMI_PER5": VarSpec("Body mass index, per 5 units", "body mass index per 5 additional units",
                        "kg/m²", "continuous"),

    # ======================================================================
    # COMORBIDITY AND PHYSIOLOGIC STATUS
    # ======================================================================
    "ASACLAS": VarSpec("ASA physical status class", "American Society of Anesthesiologists physical status class",
                       None, "ordinal"),
    "ASA_CAT": VarSpec("ASA physical status class", "ASA physical status class", None, "categorical"),
    "ASA_GE3": VarSpec("ASA class 3 or 4", "ASA physical status class 3 or 4", None, "binary"),
    "FNSTATUS2": VarSpec("Functional status", "preoperative functional status", None, "ordinal"),
    "FNSTATUS": VarSpec("Functional status", "preoperative functional status", None, "ordinal"),
    "FNSTATUS_DEPENDENT": VarSpec("Functional dependence", "partial or total preoperative functional dependence",
                                  None, "binary"),
    "DIABETES": VarSpec("Diabetes mellitus", "diabetes mellitus", None, "categorical"),
    "DIABETES_ANY": VarSpec("Diabetes mellitus", "diabetes mellitus of any type", None, "binary"),
    "SMOKE": VarSpec("Current smoker", "current smoking within one year of surgery", None, "binary"),
    "HXCOPD": VarSpec("Chronic obstructive pulmonary disease", "chronic obstructive pulmonary disease",
                      None, "binary"),
    "HXCHF": VarSpec("Congestive heart failure", "congestive heart failure within 30 days of surgery",
                     None, "binary"),
    "HYPERMED": VarSpec("Hypertension requiring medication", "medically treated hypertension", None, "binary"),
    "DIALYSIS": VarSpec("Dialysis dependence", "preoperative dialysis dependence", None, "binary"),
    # STEROID keeps its label because `02_build_cohort.py` still recodes the column (it is
    # in the ported BINARY_INPLACE roster and in FEATURE_PRETTY, and dropping either would
    # fork the two cohorts). It is excluded from every ANALYSIS and every TABLE
    # in this project: it carries an undated definition broadening and no analysis
    # needs it. It is therefore absent from SECTIONS and from every
    # contract constant below.
    "STEROID": VarSpec("Chronic corticosteroid or immunosuppressant use",
                       "chronic corticosteroid or immunosuppressant use", None, "binary"),
    "BLEEDDIS": VarSpec("Bleeding disorder", "bleeding disorder", None, "binary"),
    "DISCANCR": VarSpec("Disseminated cancer", "disseminated cancer", None, "binary"),
    "PRSEPIS": VarSpec("Preoperative sepsis", "preoperative sepsis or systemic inflammatory response",
                       None, "categorical"),
    "WNDCLAS": VarSpec("Wound class", "wound classification", None, "ordinal"),

    # ======================================================================
    # PREOPERATIVE LABORATORY VALUES
    # Raw NSQIP names are PRHCT / PRALBUM / PRINR / PRPLATE / PRPTT. The unprefixed
    # aliases exist so a downstream step that reaches for the clinical name still lands
    # on a label instead of the de-underscored fallback; both spellings render the same
    # string. The analytic parquet uses the PR-prefixed NSQIP names.
    #
    # These columns are still MATERIALISED by the ported `02_build_cohort.py`, so they keep
    # their labels. They are NOT in this project's Table 1 (SECTIONS below): the
    # coagulation panel is the parent paper's exposure, and every form
    # of the coagulopathy-burden finding is kept out of this manuscript.
    # ======================================================================
    "PRHCT": VarSpec("Preoperative hematocrit, %", "preoperative hematocrit", "%", "continuous"),
    "HCT": VarSpec("Preoperative hematocrit, %", "preoperative hematocrit", "%", "continuous"),
    "PRALBUM": VarSpec("Preoperative serum albumin, g/dL", "preoperative serum albumin", "g/dL", "continuous"),
    "ALBUMIN": VarSpec("Preoperative serum albumin, g/dL", "preoperative serum albumin", "g/dL", "continuous"),
    "PRINR": VarSpec("Preoperative international normalized ratio",
                     "preoperative international normalized ratio", None, "continuous"),
    "INR": VarSpec("Preoperative international normalized ratio",
                   "preoperative international normalized ratio", None, "continuous"),
    "PRPLATE": VarSpec("Preoperative platelet count, ×10³/µL",
                       "preoperative platelet count", "×10³/µL", "continuous"),
    "PLATELET": VarSpec("Preoperative platelet count, ×10³/µL",
                        "preoperative platelet count", "×10³/µL", "continuous"),
    "PLT": VarSpec("Preoperative platelet count, ×10³/µL",
                   "preoperative platelet count", "×10³/µL", "continuous"),
    "PRPTT": VarSpec("Preoperative partial thromboplastin time, seconds",
                     "preoperative partial thromboplastin time", "seconds", "continuous"),
    "PTT": VarSpec("Preoperative partial thromboplastin time, seconds",
                   "preoperative partial thromboplastin time", "seconds", "continuous"),
    # PRWBC and PRCREAT are deliberately NOT labelled. They exist in the warehouse but they
    # are NOT in `vars/acdf_discharge_timing.txt`, so no column by that name will ever exist
    # in `cohort_analytic.parquet`. A label for a column the pipeline cannot produce invites
    # a step to reach for it. To use them, add them to the export list FIRST
    # (which changes the bundle width `01_export_cohort.py` asserts), then re-add the labels.
    "ANEMIA": VarSpec("Preoperative anemia", "preoperative anemia", None, "binary"),

    # ======================================================================
    # COAGULOPATHY BURDEN
    # PORTED UNCHANGED and NEVER READ BY THIS PAPER. `02_build_cohort.py` still
    # materialises these columns because dropping them would fork the two cohorts;
    # `03` through `08` never read them, and the coagulopathy-burden
    # finding in every form belongs to the parent manuscript. The labels stay so the
    # builder's own schema documentation and `verify_numbers.py` can name the columns.
    # ======================================================================
    "COAG_BURDEN_N": VarSpec("Coagulopathy burden, number of abnormalities",
                             "number of accumulated coagulation abnormalities", None, "continuous"),
    "COAG_BURDEN_PER1": VarSpec("Coagulopathy burden, per additional abnormality",
                                "coagulopathy burden per additional abnormality", None, "ordinal"),
    "COAG_BURDEN_CAT": VarSpec("Coagulopathy burden", "coagulopathy burden stratum", None, "categorical"),
    "COAG_BURDEN_GE2": VarSpec("Two or more coagulation abnormalities",
                               "two or more accumulated coagulation abnormalities", None, "binary"),
    "INR_GT12": VarSpec("International normalized ratio above 1.2",
                        "preoperative international normalized ratio above 1.2", None, "binary"),
    "PLT_LT150": VarSpec("Platelet count below 150 ×10³/µL",
                         "preoperative platelet count below 150 ×10³/µL", None, "binary"),
    "PTT_GT35": VarSpec("Partial thromboplastin time above 35 seconds",
                        "preoperative partial thromboplastin time above 35 seconds", None, "binary"),

    # ======================================================================
    # PROCEDURE AND OPERATIVE CHARACTERISTICS
    # ======================================================================
    "CPT": VarSpec("Principal procedure code", "principal procedure code", None, "categorical"),
    "PRNCPTX": VarSpec("Principal procedure", "principal procedure description", None, "categorical"),
    "OTHERCPT": VarSpec("Concurrent procedure codes", "concurrent procedure codes", None, "categorical"),
    "LEVELS_N": VarSpec("Number of fusion levels", "number of fusion levels", "levels", "continuous"),
    "LEVELS_PER_LEVEL": VarSpec("Fusion levels, per level", "fusion levels per additional level",
                                "levels", "continuous"),
    "LEVELS_PER_LEVEL_CAP3": VarSpec("Fusion levels, per level up to three or more",
                                     "fusion levels per additional level up to three or more",
                                     "levels", "continuous"),
    "LEVELS_CAT": VarSpec("Fusion levels", "fusion level stratum", None, "categorical"),
    "LEVELS_GE3": VarSpec("Three or more fusion levels", "three or more fusion levels", None, "binary"),
    "ANT_INSTRUM": VarSpec("Anterior cervical instrumentation", "anterior cervical instrumentation",
                           None, "binary"),
    "OPTIME": VarSpec("Operative time, minutes", "operative time", "minutes", "continuous"),
    "SURGSPEC": VarSpec("Surgical specialty", "surgical specialty as recorded", None, "categorical"),
    "SPEC_CAT": VarSpec("Surgical specialty", "surgical specialty", None, "categorical"),
    "OPERYR": VarSpec("Operative year", "operative year", None, "categorical"),
    "PUFYEAR": VarSpec("Operative year", "ACS-NSQIP participant use file year", None, "categorical"),
    "ANESTHES": VarSpec("Anesthesia technique", "principal anesthesia technique", None, "categorical"),
    "CASETYPE": VarSpec("Case acuity", "case acuity as recorded", None, "categorical"),
    "EMERGNCY": VarSpec("Emergency case", "emergency case designation", None, "binary"),
    "ELECTSURG": VarSpec("Elective surgery flag", "the recorded elective surgery flag",
                         None, "binary"),
    "ELECTIVE": VarSpec("Elective case", "elective case", None, "binary"),
    "HTOODAY": VarSpec("Days from admission to operation", "days from hospital admission to operation",
                       "days", "continuous"),
    # NEW: the two coding-era sensitivity strata. Every timing result is reported by
    # coding era, and an EMERGNCY-defined-elective-only sensitivity is also required.
    # Both are year flags, not date fields.
    "ERA_ICD10": VarSpec("Operative year 2016 or later", "an operative year of 2016 or later",
                         None, "binary"),
    "ERA_EMERGNCY": VarSpec("Operative year 2020 or earlier", "an operative year of 2020 or earlier",
                            None, "binary"),

    # ======================================================================
    # OUTCOMES AND THEIR ASCERTAINMENT FIELDS
    # ======================================================================
    "HEMATOMA": VarSpec("Hematoma requiring reoperation",
                        "hematoma requiring unplanned reoperation within 30 days", None, "binary"),
    "HEMATOMA_CPT_ONLY": VarSpec("Hematoma, procedure-code definition",
                                 "hematoma ascertained by reoperation procedure code only", None, "binary"),
    "HEMATOMA_ICD_ONLY": VarSpec("Hematoma, diagnosis-code definition",
                                 "hematoma ascertained by reoperation diagnosis code only", None, "binary"),
    # Pre-specified code-set sensitivity definitions (the variants defined in cpt_acdf). The PRIMARY
    # outcome is HEMATOMA above, which is cpt_acdf.HEMATOMA_DEFINITION = V2. Gate 2
    # reports the V0 value beside the V2 value for the three carried-over
    # numbers that depend on the hematoma outcome.
    "HEMATOMA_V0": VarSpec("Hematoma, original code set",
                           "hematoma under the original diagnosis code set", None, "binary"),
    "HEMATOMA_V1": VarSpec("Hematoma, corrected code set without the hemorrhage arm",
                           "hematoma under the corrected diagnosis code set without the "
                           "nervous system hemorrhage arm", None, "binary"),
    # HEMATOMA_POD is the parent's column name and is KEPT. It is the hematoma entry's
    # `pod_col` in TIMED_OUTCOMES, so `02` must NOT create a second `POD_HEMATOMA` column.
    # The generator below asserts this entry matches the generated convention exactly.
    "HEMATOMA_POD": VarSpec("Postoperative day of hematoma reoperation",
                            "postoperative day of hematoma reoperation", "days", "continuous"),
    "REOPERATION1": VarSpec("Unplanned reoperation", "unplanned reoperation within 30 days", None, "binary"),
    "REOPORCPT1": VarSpec("Reoperation procedure code", "procedure code of the first unplanned reoperation",
                          None, "categorical"),
    "REOPORICD91": VarSpec("Reoperation diagnosis code",
                           "diagnosis code of the first unplanned reoperation", None, "categorical"),
    "REOPOR1ICD101": VarSpec("Reoperation diagnosis code",
                             "diagnosis code of the first unplanned reoperation", None, "categorical"),
    "REOPOR2CPT1": VarSpec("Second reoperation procedure code",
                           "procedure code of the second unplanned reoperation", None, "categorical"),
    "REOPOR2ICD91": VarSpec("Second reoperation diagnosis code",
                            "diagnosis code of the second unplanned reoperation", None, "categorical"),
    "REOPOR2ICD101": VarSpec("Second reoperation diagnosis code",
                             "diagnosis code of the second unplanned reoperation", None, "categorical"),
    "REOP_ANY": VarSpec("Any unplanned reoperation", "any unplanned reoperation within 30 days",
                        None, "binary"),
    "REINTUBATION": VarSpec("Unplanned reintubation", "unplanned reintubation", None, "binary"),
    "REINTUB": VarSpec("Unplanned reintubation", "unplanned reintubation", None, "binary"),
    "VENT48": VarSpec("Ventilator dependence beyond 48 hours",
                      "mechanical ventilation for more than 48 hours", None, "binary"),
    "FAILWEAN": VarSpec("Ventilator dependence beyond 48 hours",
                        "mechanical ventilation for more than 48 hours", None, "binary"),
    "PNEUMONIA": VarSpec("Pneumonia", "postoperative pneumonia", None, "binary"),
    "OUPNEUMO": VarSpec("Pneumonia", "postoperative pneumonia", None, "binary"),
    "AIRWAY_COMP": VarSpec("Composite airway morbidity", "composite airway morbidity", None, "binary"),
    "REOP_TRACH": VarSpec("Reoperative tracheostomy", "tracheostomy performed at unplanned reoperation",
                          None, "binary"),
    # NEW. The composite keeps
    # the parent's frozen SLOT 1 implementation so the two papers cannot disagree; this
    # wider slots-1-and-2 marker exists for the ascertainment supplement ONLY and is not in
    # AIRWAY_COMP or WINDOW_COMP.
    "REOP_TRACH_SLOT12": VarSpec("Reoperative tracheostomy, first or second reoperation",
                                 "tracheostomy performed at the first or second unplanned reoperation",
                                 None, "binary"),
    # NEW: the safe-discharge composite. Hematoma OR the three-component
    # airway composite. Pneumonia and unplanned readmission are deliberately EXCLUDED.
    "WINDOW_COMP": VarSpec("Composite hematoma or airway outcome",
                           "the composite of hematoma requiring reoperation and airway morbidity",
                           None, "binary"),
    "READMIT_UNPLANNED": VarSpec("Unplanned readmission", "unplanned readmission within 30 days",
                                 None, "binary"),
    "UNPLANNEDREADMISSION1": VarSpec("Unplanned readmission", "unplanned readmission within 30 days",
                                     None, "binary"),
    "READMISSION1": VarSpec("Any readmission", "any readmission within 30 days", None, "binary"),
    "READMRELICD91": VarSpec("Readmission diagnosis code", "diagnosis code recorded for the readmission",
                             None, "categorical"),
    "READMRELICD101": VarSpec("Readmission diagnosis code", "diagnosis code recorded for the readmission",
                              None, "categorical"),
    "READMSUSPREASON1": VarSpec("Suspected reason for readmission",
                                "the suspected reason recorded for the readmission",
                                None, "categorical"),
    "MORT30": VarSpec("Death within 30 days", "death within 30 days of surgery", None, "binary"),
    "MORT_ANY_RECORDED": VarSpec("Death recorded in any field",
                                 "death recorded in any death field", None, "binary"),
    "YRDEATH": VarSpec("Year of death", "year of death", None, "categorical"),
    "DYSPHAGIA_READMIT": VarSpec("Dysphagia-coded readmission",
                                 "readmission carrying a dysphagia diagnosis code", None, "binary"),
    "DYSPHAGIA_REOP": VarSpec("Dysphagia-coded reoperation",
                              "reoperation carrying a dysphagia diagnosis code", None, "binary"),
    "DYSPHAGIA_ANY": VarSpec("Dysphagia-coded readmission or reoperation",
                             "readmission or reoperation carrying a dysphagia diagnosis code",
                             None, "binary"),

    # ======================================================================
    # RAW NSQIP DAYS-TO-EVENT FIELDS
    #
    # These are the raw fields every `POD_*` column is taken from, and they are what
    # supplement S2 tabulates: coverage, and in particular MISSINGNESS CONDITIONAL ON THE
    # EVENT FLAG BEING POSITIVE, which is the measurement that matters and is unmeasured
    # today. The database-wide percentages quoted earlier for these fields are EVENT
    # RATES, not missingness rates; the manuscript must not confuse the two.
    #
    # RETOR2PODAYS and READMPODAYS2 are here because only the FIRST event of each type
    # carries a timing field. All analyses are
    # time-to-first-event; the second-occurrence day columns exist so `03` can state how
    # many patients have a second event whose timing is only partially observed.
    # ======================================================================
    "RETORPODAYS": VarSpec("Days from operation to reoperation", "days from operation to unplanned reoperation",
                           "days", "continuous"),
    "RETOR2PODAYS": VarSpec("Days from operation to second reoperation",
                            "days from operation to the second unplanned reoperation", "days", "continuous"),
    "READMPODAYS1": VarSpec("Days from operation to readmission", "days from operation to readmission",
                            "days", "continuous"),
    "READMPODAYS2": VarSpec("Days from operation to second readmission",
                            "days from operation to the second readmission", "days", "continuous"),
    "DREINTUB": VarSpec("Days from operation to reintubation",
                        "days from operation to unplanned reintubation", "days", "continuous"),
    "DFAILWEAN": VarSpec("Days from operation to ventilator dependence beyond 48 hours",
                         "days from operation to ventilator dependence beyond 48 hours",
                         "days", "continuous"),
    "DOUPNEUMO": VarSpec("Days from operation to pneumonia", "days from operation to pneumonia",
                         "days", "continuous"),
    "DOPERTOD": VarSpec("Days from operation to death", "days from operation to death", "days", "continuous"),
    # STILLINHOSP is the NSQIP flag for a patient still admitted at the close of the
    # 30-day window. It bears directly on this paper: a patient who never left has no
    # post-discharge period, so the post-discharge fraction's denominator has to account
    # for them. It goes in supplement S2 with its count.
    "STILLINHOSP": VarSpec("Still in hospital beyond 30 days",
                           "still in hospital beyond 30 days after the operation", None, "binary"),

    # ======================================================================
    # POSTOPERATIVE LENGTH OF STAY, DISCHARGE AND DISPOSITION
    #
    # `DOPTODIS` (days from operation to discharge) is the TRUE postoperative LOS column
    # and is PRIMARY. `LOS_POSTOP` is the derived analytic
    # column and `02_build_cohort.py` must source it from `DOPTODIS`, NEVER from
    # TOTHLOS - HTOODAY and NEVER from INOUT. The subtraction is materialised separately
    # as LOS_POSTOP_CHECK and is a CROSS-CHECK only: the parent measured 204 disagreements
    # on 112,523 analytic rows where both are present, every one of them the subtraction
    # reading exactly ONE DAY LONGER.
    #
    # `INOUT` is the BILLING label. It is NOT a discharge measure and it is NOT a length of
    # stay. Its display labels say "billing" on purpose, so a reader of a table
    # cannot mistake it for time in hospital. Re-deriving "outpatient" from it is the
    # measurement error this paper is about. The parent project bans the
    # column outright; this project needs it and lifts the ban for the critique only.
    # ======================================================================
    "TOTHLOS": VarSpec("Total hospital length of stay, days", "total hospital length of stay",
                       "days", "continuous"),
    "DOPTODIS": VarSpec("Postoperative length of stay, days", "postoperative length of stay",
                        "days", "continuous"),
    "LOS_POSTOP": VarSpec("Postoperative length of stay, days", "postoperative length of stay",
                          "days", "continuous"),
    "LOS_POSTOP_CHECK": VarSpec("Postoperative length of stay, cross-check, days",
                                "the cross-check postoperative length of stay, total hospital "
                                "length of stay minus days from admission to operation",
                                "days", "continuous"),
    "LOS_POSTOP_AGREE": VarSpec("Length of stay fields agree",
                                "agreement between the two postoperative length of stay fields",
                                None, "binary"),
    "LOS_CAT": VarSpec("Discharge timing", "discharge timing stratum", None, "categorical"),
    "SAMEDAY_DISCH": VarSpec("Same day discharge", "discharge on the day of the operation",
                             None, "binary"),
    "DISCH_BY_POD1": VarSpec("Discharge by postoperative day 1",
                             "discharge by postoperative day 1", None, "binary"),
    "INOUT": VarSpec("Billing setting, inpatient or outpatient",
                     "the recorded inpatient or outpatient billing setting", None, "categorical"),
    "INOUT_OUTPT": VarSpec("Outpatient billing setting", "an outpatient billing setting",
                           None, "binary"),
    "DISCHDEST": VarSpec("Discharge destination", "discharge destination as recorded", None, "categorical"),
    "DEST_CAT": VarSpec("Discharge destination", "discharge destination", None, "categorical"),
    "NONHOME_DISCH": VarSpec("Non-home discharge", "discharge to a destination other than home",
                             None, "binary"),

    # ======================================================================
    # TIMING COLUMNS THAT ARE NOT PART OF THE EIGHT-OUTCOME ROSTER
    #
    # POD_TRACH / POD_TRACH12 support the ascertainment supplement's slot-1 versus
    # slots-1-and-2 comparison. POD_WINDOW is the safe-discharge
    # composite's day and is what `06_window.py` reads; the composite has no
    # POSTDISCH_ or TIMED_BOTH_ column because it is not one of the eight timed
    # outcomes in the denominator ladder.
    # ======================================================================
    "POD_TRACH": VarSpec("Postoperative day of reoperative tracheostomy",
                         "postoperative day of reoperative tracheostomy", "days", "continuous"),
    "POD_TRACH12": VarSpec("Postoperative day of reoperative tracheostomy, first or second reoperation",
                           "postoperative day of reoperative tracheostomy at the first or second "
                           "unplanned reoperation", "days", "continuous"),
    "POD_WINDOW": VarSpec("Postoperative day of the composite hematoma or airway outcome",
                          "postoperative day of the composite hematoma or airway outcome",
                          "days", "continuous"),

    # ======================================================================
    # COHORT-FLOW AND DENOMINATOR FLAGS (Figure 1, and the analytic-cohort denominators)
    # ======================================================================
    "EXCL_CORPECTOMY": VarSpec("Concurrent corpectomy", "concurrent corpectomy", None, "binary"),
    "EXCL_POSTERIOR": VarSpec("Concurrent posterior spinal decompression or instrumentation",
                              "concurrent posterior spinal decompression or instrumentation",
                              None, "binary"),
    "EXCL_ARTHROPLASTY": VarSpec("Concurrent cervical disc arthroplasty",
                                 "concurrent cervical disc arthroplasty", None, "binary"),
    "EXCL_ASA5": VarSpec("ASA class 5", "ASA physical status class 5", None, "binary"),
    "EXCL_ASA_UNASSIGNED": VarSpec("ASA class not assigned", "an unassigned ASA class",
                                   None, "binary"),
    "EXCL_PRESEPSIS": VarSpec("Preoperative sepsis", "preoperative sepsis", None, "binary"),
    "STEP_ELECTIVE": VarSpec("Elective cohort", "the elective cohort", None, "binary"),
    "STEP_EXCLUSIONS": VarSpec("Cohort after procedure exclusions",
                               "the cohort after the procedure exclusions", None, "binary"),
    "STEP_ANALYTIC": VarSpec("Analytic cohort", "the analytic cohort", None, "binary"),
    "MODEL_COMPLETE": VarSpec("Complete for the ported completeness roster",
                              "complete data on every column in the ported completeness roster",
                              None, "binary"),
    "LAB_COMPLETE": VarSpec("Complete coagulation panel",
                            "a complete preoperative coagulation panel", None, "binary"),
    "INR_MEASURED": VarSpec("International normalized ratio measured",
                            "a measured preoperative international normalized ratio", None, "binary"),
}


# ---------------------------------------------------------------------------
# GENERATED LABELS: the three column families keyed by TIMED_OUTCOMES.
#
# Eight outcomes times three families is 24 names. Typing them out three times is how the
# `POD_` spelling of one outcome and the `POSTDISCH_` spelling of the same outcome end up
# describing different things. They are generated from one roster instead.
#
# An entry already present in the literal above (HEMATOMA_POD) is NOT overwritten: it must
# MATCH the generated spec exactly, or this raises. That is what proves the parent's
# ported HEMATOMA_POD label and this project's generation convention agree.
# ---------------------------------------------------------------------------
def _timed_family_specs(o: TimedOutcome) -> Tuple[Tuple[str, VarSpec], ...]:
    cap = o.noun[0].upper() + o.noun[1:]
    return (
        (o.pod_col, VarSpec(
            f"Postoperative day of {o.noun}",
            f"postoperative day of {o.noun}",
            "days", "continuous")),
        (f"POSTDISCH_{o.suffix}", VarSpec(
            f"{cap} after discharge",
            f"{o.noun} occurring after that patient's own discharge",
            None, "binary")),
        (f"TIMED_BOTH_{o.suffix}", VarSpec(
            f"{cap} with event day and length of stay both recorded",
            f"{o.noun} with both the day of the event and the postoperative length of "
            f"stay recorded",
            None, "binary")),
    )


for _o in TIMED_OUTCOMES:
    for _col, _generated in _timed_family_specs(_o):
        _existing = LABELS.get(_col)
        if _existing is None:
            LABELS[_col] = _generated
        elif _existing != _generated:
            raise ValueError(
                f"LABELS[{_col!r}] was written by hand as {_existing!r} but the "
                f"TIMED_OUTCOMES generator produces {_generated!r}. Edit one of them; "
                f"do not let the roster and the label drift apart."
            )
del _o, _col, _generated, _existing


# ---------------------------------------------------------------------------
# Lookup helpers (carried over unchanged)
# ---------------------------------------------------------------------------
def short(code: str) -> str:
    """Reader-facing table stub for a variable. Raises KeyError if the key is unknown."""
    return LABELS[code].short_label


def prose(code: str) -> str:
    """Lowercase mid-sentence form for a variable. Raises KeyError if the key is unknown."""
    return LABELS[code].prose_label


def unit(code: str) -> str:
    """Unit string, or the empty string when the variable is unitless."""
    u = LABELS[code].units
    return u if u is not None else ""


def var_type(code: str) -> str:
    """One of "continuous" | "binary" | "categorical" | "ordinal"."""
    return LABELS[code].var_type


# ---------------------------------------------------------------------------
# Dash constants.
#
# EM_DASH is written as an escape ON PURPOSE: no file in this project may contain a
# literal U+2014, so a repo-wide grep for the character stays clean and the ban is
# enforceable by search. Compare against EM_DASH, never against a pasted character.
# ---------------------------------------------------------------------------
EN_DASH = "–"        # U+2013. The range separator. CORRECT, do not strip it.
EM_DASH = "\u2014"   # U+2014. BANNED in rendered text.


# A snake_case identifier is any alphanumeric-underscore-alphanumeric run. Matching
# ANYWHERE in the string (not token-by-token) is the point: the most likely leak form is a
# parenthesised or bracketed column name inside an otherwise clean axis label, for example
# "Discharge (los_cat)". There is no punctuation escape hatch, by design.
_SNAKE_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]")


def has_snake_case(text) -> bool:
    """True if `text` still contains an unrendered snake_case identifier, anywhere.

    Guard for `07_figures.py` and `08_generate_manuscript.py`: assert this is False for
    every string written into a figure, a table CSV, or the manuscript. The no-snake_case
    rule depends on it.

    True:  "los_cat", "(levels_n)", "Timing (pod_reintub)", "[age_cat]",
           "Share (postdisch_hematoma)", "x=los_postop"
    False: "Discharge timing", "Age, years", "ASA class 3 or 4", "1", "", "_", "__"
    """
    return bool(_SNAKE_RE.search(str(text)))


def has_em_dash(text) -> bool:
    """True if `text` contains an EM-dash (U+2014). Companion to has_snake_case().

    EN_DASH (U+2013) is fine and expected inside `fmt_iqr` / `fmt_or_ci` / `fmt_rr_ci`
    output and inside `AGE_CAT_PRETTY`; this function does not flag it.
    """
    return EM_DASH in str(text)


class RenderabilityError(ValueError):
    """Raised by assert_renderable() when a string is not fit for rendered output."""


def assert_renderable(text, where: str = "") -> str:
    """Raise unless `text` is fit to be written into a figure, a table CSV, or the docx.

    THE write-time gate for the no-em-dash and no-snake_case rules. Call it on every
    string immediately before it reaches `outputs/v1/*.docx`, `figures/*` or
    `tables-csv/*`; it returns the string so it can be used inline:

        ax.set_xlabel(assert_renderable(pretty("LOS_POSTOP"), where="Figure 4 x axis"))

    Raises RenderabilityError (a ValueError) rather than asserting, so the guard survives
    `python3 -O`. `where` is a free-text location hint that lands in the message.
    """
    s = str(text)
    loc = f" at {where}" if where else ""
    if has_em_dash(s):
        raise RenderabilityError(
            f"EM-dash (U+2014) in rendered text{loc}: {s!r}. "
            f"Use EN_DASH (U+2013) for ranges, or rewrite the sentence."
        )
    if has_snake_case(s):
        raise RenderabilityError(
            f"snake_case leak in rendered text{loc}: {s!r}. "
            f"Route the name through variable_dict.pretty() or pretty_level()."
        )
    return s


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
#
# Every table cell and every manuscript number goes through one of these. They are the
# reason a rounding convention cannot drift between Table 3, Figure 2 and the Results
# paragraph that quotes them.
#
# EN_DASH (U+2013) is the range separator throughout. It is not the banned dash.
# ---------------------------------------------------------------------------
def fmt_count(n) -> str:
    """Integer count with thousands separators: 112642 -> "112,642"."""
    return f"{int(n):,}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """A percentage already expressed on the 0-100 scale, at a stated precision.

    fmt_pct(0.477, 3) -> "0.477%"   fmt_pct(17.8) -> "17.8%"
    """
    return f"{value:.{decimals}f}%"


def fmt_n_pct(n, total, decimals: int = 1) -> str:
    """Count and percentage: fmt_n_pct(541, 112642) -> "541 (0.5%)".

    A zero denominator yields 0.0% rather than raising, because Table 1 legitimately has
    empty strata. Use fmt_n_pct_denom() wherever the denominator is not obvious from the
    surrounding column header.
    """
    pct = (n / total * 100.0) if total else 0.0
    return f"{int(n):,} ({pct:.{decimals}f}%)"


def fmt_n_pct_denom(n, total, decimals: int = 1) -> str:
    """Count, explicit denominator, and percentage.

    fmt_n_pct_denom(541, 112642) -> "541/112,642 (0.5%)"

    THIS is the default form in this project, not the bare fmt_n_pct(). Rule: no
    proportion is reported without stating the population it is a proportion
    of, and in particular no post-discharge fraction may be printed without row 4 of its
    own denominator ladder beside it. This project carries several distinct denominators:

      * cohort ladder: import it from `cpt_acdf.COHORT_LADDER_N` (Gate B, primary) or
        `cpt_acdf.COHORT_LADDER_N_PLAN` (Gate A, reconciliation only). Never retype one.
      * the `INOUT = 'Outpatient'` subset: measured by `05_inout.py` on the ANALYTIC
        cohort, with the identified-cohort figure published beside
        it for reconciliation against the carried-over 47,713.
      * one analysis denominator PER TIMED OUTCOME: row 4 of the denominator ladder,
        "events with BOTH the days-to-event field and the postoperative length of stay
        present". It is `TIMED_BOTH_<X>`, never "all events".
    """
    pct = (n / total * 100.0) if total else 0.0
    return f"{int(n):,}/{int(total):,} ({pct:.{decimals}f}%)"


def fmt_mean_sd(mean: float, sd: float, decimals: int = 1) -> str:
    """Mean and standard deviation: fmt_mean_sd(54.2, 11.8) -> "54.2 (11.8)"."""
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def fmt_iqr(median: float, q1: float, q3: float, decimals: int = 1) -> str:
    """Median and interquartile range, EN-dash separated.

    fmt_iqr(2, 1, 4, 0) -> "2 (1–4)"
    """
    return f"{median:.{decimals}f} ({q1:.{decimals}f}{EN_DASH}{q3:.{decimals}f})"


# House alias. Some downstream code reaches for the clinical name; both are this function,
# so the rounding convention cannot fork.
fmt_median_iqr = fmt_iqr


def fmt_or_ci(or_: float, lo: float, hi: float, decimals: int = 2) -> str:
    """Odds ratio with 95% confidence interval, EN-dash separated.

    fmt_or_ci(2.53, 1.45, 4.40) -> "2.53 (1.45–4.40)"
    """
    return f"{or_:.{decimals}f} ({lo:.{decimals}f}{EN_DASH}{hi:.{decimals}f})"


def fmt_rr_ci(rr: float, lo: float, hi: float, decimals: int = 2) -> str:
    """Risk ratio with 95% confidence interval. Same body as fmt_or_ci; the separate name
    keeps a subdistribution-hazard or risk-ratio cell from being silently relabelled an
    odds ratio in the both-ways comparison table."""
    return f"{rr:.{decimals}f} ({lo:.{decimals}f}{EN_DASH}{hi:.{decimals}f})"


def fmt_ci(lo: float, hi: float, decimals: int = 2) -> str:
    """A bare interval when the point estimate is in its own column.

    fmt_ci(1.45, 4.40) -> "1.45–4.40"
    """
    return f"{lo:.{decimals}f}{EN_DASH}{hi:.{decimals}f}"


def fmt_beta_se(beta: float, se: float, decimals: int = 3) -> str:
    """Coefficient and standard error: fmt_beta_se(0.928, 0.283) -> "0.928 (0.283)"."""
    return f"{beta:.{decimals}f} ({se:.{decimals}f})"


def fmt_smd(d: float, decimals: int = 3) -> str:
    """Absolute standardized mean difference, for any balance table."""
    return f"{abs(d):.{decimals}f}"


P_THRESHOLD = 0.001
"""Below this, a p-value is reported as an inequality rather than a rounded number."""


def fmt_p(p: float, spaced: bool = True) -> str:
    """A p-value. Below 0.001 it reports "< 0.001"; it can never render "0.000".

    fmt_p(0.001)   -> "0.001"
    fmt_p(0.0009)  -> "< 0.001"
    fmt_p(0.0)     -> "< 0.001"
    fmt_p(0.371)   -> "0.371"

    `spaced=False` gives the compact "<0.001" for a narrow table column. The spaced form
    is the project default; pick one per table and stay with it.
    """
    if p < P_THRESHOLD:
        return "< 0.001" if spaced else "<0.001"
    out = f"{p:.3f}"
    if out == "0.000":  # unreachable given the branch above; belt and braces
        return "< 0.001" if spaced else "<0.001"
    return out


# ---------------------------------------------------------------------------
# DISPLAY DICTS
#
# Every model-feature name and every categorical level maps to human-readable text here.
# Figures, tables and the manuscript route ALL user-visible labels through pretty() and
# pretty_level(); no snake_case token and no raw NSQIP spelling may reach rendered text.
#
# Keys on the LEFT are the values as they appear IN THE DATA (NSQIP spelling, or the
# ASCII level strings 02_build_cohort.py writes). Display text is on the RIGHT. Never put
# a display spelling on the left as an alias: level lookup is already case-insensitive.
# ---------------------------------------------------------------------------

# FEATURE_PRETTY is PORTED FROM THE PARENT AND KEPT DELIBERATELY, even though this project
# fits no hematoma model. Two reasons:
#   1. The ported `02_build_cohort.py` asserts at import that every column it recodes in
#      place from Yes/No text to 0/1 is named here. Deleting the dict, or dropping a key
#      from it, changes which raw columns get recoded and FORKS the two cohorts.
#   2. It keeps `pretty()` resolving a covariate name that appears in a ported schema note.
# It is NOT an analysis roster for this project. The six adjusted odds ratios it was
# written for belong to the parent manuscript and are not analysed here.
# Values MUST equal LABELS[key].short_label; the reconciliation loop at the bottom of this
# file RAISES on any drift rather than silently overwriting.
FEATURE_PRETTY: Dict[str, str] = {
    "BMI_PER5": "Body mass index, per 5 units",
    "MALE": "Male sex",
    "LEVELS_PER_LEVEL": "Fusion levels, per level",
    "ASA_GE3": "ASA class 3 or 4",
    "AGE_PER10": "Age, per decade",
    "HYPERMED": "Hypertension requiring medication",
    "BLEEDDIS": "Bleeding disorder",
    "DIABETES_ANY": "Diabetes mellitus",
    "SMOKE": "Current smoker",
    "HXCOPD": "Chronic obstructive pulmonary disease",
    "HXCHF": "Congestive heart failure",
    "DIALYSIS": "Dialysis dependence",
    "FNSTATUS_DEPENDENT": "Functional dependence",
    "STEROID": "Chronic corticosteroid or immunosuppressant use",
    "ANEMIA": "Preoperative anemia",
    "ANT_INSTRUM": "Anterior cervical instrumentation",
    "OPTIME": "Operative time, minutes",
    "SPEC_CAT": "Surgical specialty",
    "OPERYR": "Operative year",
    "COAG_BURDEN_CAT": "Coagulopathy burden",
    "COAG_BURDEN_PER1": "Coagulopathy burden, per additional abnormality",
    "COAG_BURDEN_GE2": "Two or more coagulation abnormalities",
    "INR_GT12": "International normalized ratio above 1.2",
    "PLT_LT150": "Platelet count below 150 ×10³/µL",
    "PTT_GT35": "Partial thromboplastin time above 35 seconds",
}

SEX_PRETTY: Dict[str, str] = {
    "female": "Female",
    "male": "Male",
    "non-binary": "Non-binary",
    "intersex": "Intersex",
    "unknown": "Unknown",
}

MALE_PRETTY: Dict[str, str] = {
    "1": "Male",
    "0": "Female",
}

# ASACLAS as NSQIP spells it, plus the bare digits in case a step stores ASA numerically.
ASA_PRETTY: Dict[str, str] = {
    "1-No Disturb": "Class 1",
    "2-Mild Disturb": "Class 2",
    "3-Severe Disturb": "Class 3",
    "4-Life Threat": "Class 4",
    "5-Moribund": "Class 5",
    "None assigned": "Not assigned",
    "1": "Class 1",
    "2": "Class 2",
    "3": "Class 3",
    "4": "Class 4",
    "5": "Class 5",
}

ASA_CAT_ORDER: List[str] = ["1-2", ">=3"]
ASA_CAT_PRETTY: Dict[str, str] = {
    "1-2": "Class 1 or 2",
    ">=3": "Class 3 or 4",
}

# NEW. The locked age stratifier. The cut points are the conventional
# adult surgical bands and were fixed a priori, before any curve was fitted.
# The LEFT-hand strings are ASCII and are exactly what `02_build_cohort.py` writes into
# `AGE_CAT` in the parquet. The EN-dash display form lives only on the right.
AGE_CAT_ORDER: List[str] = ["18-49", "50-64", "65-74", ">=75"]
AGE_CAT_PRETTY: Dict[str, str] = {
    "18-49": f"18{EN_DASH}49 years",
    "50-64": f"50{EN_DASH}64 years",
    "65-74": f"65{EN_DASH}74 years",
    ">=75": "75 years or older",
}

# NEW. Discharge timing, the Table 1 stratifier and this paper's exposure of interest.
# It is derived from LOS_POSTOP (that is, from `DOPTODIS`) and NEVER from `INOUT`.
# `02_build_cohort.py` writes the empty string where LOS_POSTOP is missing, and an empty
# string is NOT a level here: `pretty_level("LOS_CAT", "")` returns an empty string rather
# than MISSING_LABEL, so `05_inout.py` must route a missing stratum through MISSING_LABEL
# explicitly or drop it with its count stated. Do not add "" as a key: the ordered-level
# invariant below requires the dict and the ORDER list to match exactly.
LOS_CAT_ORDER: List[str] = ["0", "1", ">=2"]
LOS_CAT_PRETTY: Dict[str, str] = {
    "0": "Same day",
    "1": "1 night",
    ">=2": "2 or more nights",
}

# NEW. `INOUT` is the BILLING setting and is never a discharge measure or a length of
# stay. The display strings say "billing" on purpose: a reader of Table 1 or
# Table 4 must not be able to read an `INOUT` row as time in hospital. The whole point of
# the billing-label critique is that the two disagree in the large majority of cases labelled
# Outpatient. Two levels only, verified read-only against the warehouse on 2026-07-31
# (`Inpatient` 6,457,523 rows, `Outpatient` 5,357,395 rows, plus ONE SQL NULL row
# database-wide, which is not in this cohort but must not crash the recode).
INOUT_ORDER: List[str] = ["Inpatient", "Outpatient"]
INOUT_PRETTY: Dict[str, str] = {
    "Inpatient": "Inpatient billing setting",
    "Outpatient": "Outpatient billing setting",
}

FNSTATUS_PRETTY: Dict[str, str] = {
    "Independent": "Independent",
    "Partially Dependent": "Partially dependent",
    "Totally Dependent": "Totally dependent",
    "Unknown": "Unknown",
}

# NSQIP writes DIABETES in upper case ("NO") while most other flags are title case
# ("No"). Lookup folds case, so only one spelling is needed on the left.
DIABETES_PRETTY: Dict[str, str] = {
    "NO": "None",
    "NON-INSULIN": "Non-insulin dependent",
    "INSULIN": "Insulin dependent",
}

ETHNICITY_PRETTY: Dict[str, str] = {
    "Yes": "Hispanic or Latino",
    "No": "Not Hispanic or Latino",
    "Unknown": "Unknown or not reported",
    "1": "Hispanic or Latino",
    "0": "Not Hispanic or Latino",
}

RACE_CAT_ORDER: List[str] = ["White", "Black", "Hispanic", "Asian", "Other", "Unknown"]
RACE_CAT_PRETTY: Dict[str, str] = {
    "White": "White",
    "Black": "Black or African American",
    "Hispanic": "Hispanic or Latino",
    "Asian": "Asian",
    "Other": "Other race",
    "Unknown": "Unknown or not reported",
}

PRSEPIS_PRETTY: Dict[str, str] = {
    "None": "None",
    "SIRS": "Systemic inflammatory response syndrome",
    "Sepsis": "Sepsis",
    "Septic Shock": "Septic shock",
}

CASETYPE_PRETTY: Dict[str, str] = {
    "Elective": "Elective",
    "Urgent": "Urgent",
    "Emergent": "Emergent",
}

SURGSPEC_PRETTY: Dict[str, str] = {
    "Neurosurgery": "Neurosurgery",
    "Orthopedics": "Orthopaedic surgery",
    "General Surgery": "General surgery",
    "Otolaryngology (ENT)": "Otolaryngology",
    "Plastics": "Plastic surgery",
    "Thoracic": "Thoracic surgery",
    "Vascular": "Vascular surgery",
    "Urology": "Urology",
    "Gynecology": "Gynecology",
    "Obstetrics": "Obstetrics",
    "Cardiac Surgery": "Cardiac surgery",
    "Interventional Radiologist": "Interventional radiology",
    "Other": "Other specialty",
    "Unknown": "Unknown specialty",
}

SPEC_CAT_ORDER: List[str] = ["Neurosurgery", "Orthopaedic", "Other"]
SPEC_CAT_PRETTY: Dict[str, str] = {
    "Neurosurgery": "Neurosurgery",
    "Orthopaedic": "Orthopaedic surgery",
    "Other": "Other specialty",
}

# The fourteen raw `DISCHDEST` levels. The vocabulary swaps COMPLETELY at 2021, so both
# vintages are here. See DISCHDEST_LEVEL_MAP below for the collapse to Home / Facility /
# Other, which is the part that must never be done by substring matching.
DISCHDEST_PRETTY: Dict[str, str] = {
    "Home": "Home",
    "Home/Permanent Residence": "Home",
    "Facility Which was Home": "Facility that was home before surgery",
    "Skilled Care, Not Home": "Skilled nursing facility",
    "Unskilled Facility Not Home": "Unskilled facility",
    "Rehab": "Rehabilitation facility",
    "Other Facility": "Other facility",
    "Separate Acute Care": "Separate acute care hospital",
    "Acute Care Hospital": "Acute care hospital",
    "Multi-level Senior Community": "Multi-level senior community",
    "Hospice": "Hospice",
    "Against Medical Advice (AMA)": "Against medical advice",
    "Expired": "Died before discharge",
    "Unknown": "Unknown or not reported",
}

DEST_CAT_ORDER: List[str] = ["Home", "Facility", "Other"]
DEST_CAT_PRETTY: Dict[str, str] = {
    "Home": "Home",
    "Facility": "Rehabilitation or skilled nursing facility",
    "Other": "Other or unknown destination",
}

# Fusion-level and coagulopathy-burden strata. The level strings on the left are ASCII and
# LOCKED: 02_build_cohort.py must write exactly these into the analytic parquet so the
# keys survive a CSV round-trip. The ">=" display forms live on the right only.
LEVELS_CAT_ORDER: List[str] = ["1", "2", ">=3"]
LEVELS_CAT_PRETTY: Dict[str, str] = {
    "1": "1 level",
    "2": "2 levels",
    ">=3": "3 or more levels",
}

COAG_BURDEN_CAT_ORDER: List[str] = ["0", "1", ">=2"]
COAG_BURDEN_CAT_PRETTY: Dict[str, str] = {
    "0": "No abnormality",
    "1": "One abnormality",
    ">=2": "Two or more abnormalities",
}

# NSQIP stores its complication flags as descriptive text rather than Yes/No.
COMPLICATION_PRETTY: Dict[str, str] = {
    "No Complication": "No",
    "Unplanned Intubation": "Yes",
    "Pneumonia": "Yes",
    "On Ventilator greater than 48 Hours": "Yes",
}

# The fallback for any binary variable with no dedicated dict of its own.
YESNO_PRETTY: Dict[str, str] = {
    "1": "Yes",
    "0": "No",
}

# The three pre-specified residual-risk thresholds for the safe discharge window, as
# display text. Keyed by str(threshold) so the dict has the standard str -> str shape and
# is swept by the invariants below. `06_window.py` uses threshold_label(); nothing types
# "1.0%" or "0.50%" by hand.
RISK_THRESHOLD_PRETTY: Dict[str, str] = {
    "0.01": "1%",
    "0.005": "0.5%",
    "0.0025": "0.25%",
}

# variable name -> the level dict that renders its categories.
LEVEL_PRETTY: Dict[str, Dict[str, str]] = {
    "SEX": SEX_PRETTY,
    "MALE": MALE_PRETTY,
    "ASACLAS": ASA_PRETTY,
    "ASA_CAT": ASA_CAT_PRETTY,
    "AGE_CAT": AGE_CAT_PRETTY,
    "LOS_CAT": LOS_CAT_PRETTY,
    "INOUT": INOUT_PRETTY,
    "FNSTATUS2": FNSTATUS_PRETTY,
    "FNSTATUS": FNSTATUS_PRETTY,
    "DIABETES": DIABETES_PRETTY,
    "ETHNICITY_HISPANIC": ETHNICITY_PRETTY,
    "RACE_CAT": RACE_CAT_PRETTY,
    "PRSEPIS": PRSEPIS_PRETTY,
    "CASETYPE": CASETYPE_PRETTY,
    "SURGSPEC": SURGSPEC_PRETTY,
    "SPEC_CAT": SPEC_CAT_PRETTY,
    "DISCHDEST": DISCHDEST_PRETTY,
    "DEST_CAT": DEST_CAT_PRETTY,
    "LEVELS_CAT": LEVELS_CAT_PRETTY,
    "COAG_BURDEN_CAT": COAG_BURDEN_CAT_PRETTY,
    "REINTUB": COMPLICATION_PRETTY,
    "FAILWEAN": COMPLICATION_PRETTY,
    "OUPNEUMO": COMPLICATION_PRETTY,
}

# Registry of every display dict, so the self-test and any downstream guard can sweep them
# all without naming them one by one. Add new dicts here when you add them above.
ALL_PRETTY_DICTS: Dict[str, Dict[str, str]] = {
    "FEATURE_PRETTY": FEATURE_PRETTY,
    "SEX_PRETTY": SEX_PRETTY,
    "MALE_PRETTY": MALE_PRETTY,
    "ASA_PRETTY": ASA_PRETTY,
    "ASA_CAT_PRETTY": ASA_CAT_PRETTY,
    "AGE_CAT_PRETTY": AGE_CAT_PRETTY,
    "LOS_CAT_PRETTY": LOS_CAT_PRETTY,
    "INOUT_PRETTY": INOUT_PRETTY,
    "FNSTATUS_PRETTY": FNSTATUS_PRETTY,
    "DIABETES_PRETTY": DIABETES_PRETTY,
    "ETHNICITY_PRETTY": ETHNICITY_PRETTY,
    "RACE_CAT_PRETTY": RACE_CAT_PRETTY,
    "PRSEPIS_PRETTY": PRSEPIS_PRETTY,
    "CASETYPE_PRETTY": CASETYPE_PRETTY,
    "SURGSPEC_PRETTY": SURGSPEC_PRETTY,
    "SPEC_CAT_PRETTY": SPEC_CAT_PRETTY,
    "DISCHDEST_PRETTY": DISCHDEST_PRETTY,
    "DEST_CAT_PRETTY": DEST_CAT_PRETTY,
    "LEVELS_CAT_PRETTY": LEVELS_CAT_PRETTY,
    "COAG_BURDEN_CAT_PRETTY": COAG_BURDEN_CAT_PRETTY,
    "COMPLICATION_PRETTY": COMPLICATION_PRETTY,
    "YESNO_PRETTY": YESNO_PRETTY,
    "RISK_THRESHOLD_PRETTY": RISK_THRESHOLD_PRETTY,
}


def pretty(name: str) -> str:
    """Display text for a variable or model-feature name.

    THE entry point for figure axis labels, legend entries, table stubs and manuscript
    prose. Resolution order: LABELS.short_label -> FEATURE_PRETTY -> a de-underscored
    fallback.

    `LABELS` is authoritative, so `pretty(k) == short(k)` for every key in `LABELS` by
    construction. Table 1 (built from SECTIONS + short()) and every figure legend (built
    from pretty()) therefore cannot label the same covariate two different ways. Every
    FEATURE_PRETTY key is required to exist in LABELS with an identical string; the
    reconciliation loop at the bottom of this file raises if that ever stops being true.

    The de-underscored fallback exists so a forgotten key degrades to readable text rather
    than leaking a snake_case token. It is a safety net, not a place to rely on: add the
    key to LABELS properly.
    """
    if name in LABELS:
        return LABELS[name].short_label
    if name in FEATURE_PRETTY:
        return FEATURE_PRETTY[name]
    return name.replace("_", " ").strip().capitalize()


# Display text for a NaN / None level. One label, used everywhere, so a missing category
# never renders as "nan" in a table stub or a figure legend.
MISSING_LABEL = "Not recorded"

# Boolean-ish level tokens, lowercased. Used to fold 1 / 1.0 / True / "1" / "1.0" /
# "True" / "Yes" onto the canonical "1" key, and the false side onto "0", before lookup.
_BOOL_TRUE_TOKENS = frozenset({"1", "1.0", "true", "yes", "y", "t"})
_BOOL_FALSE_TOKENS = frozenset({"0", "0.0", "false", "no", "n", "f"})


def is_missing_level(level) -> bool:
    """True for None / NaN / pd.NA. Strings are NEVER missing.

    The string exclusion is deliberate and load-bearing: `PRSEPIS` has a real "None" level
    meaning no sepsis, and `DISCHDEST` and `INOUT` have real text levels. Blanket-recoding
    the string "None" to missing would corrupt all three.
    """
    if level is None:
        return True
    if isinstance(level, str):
        return False
    try:
        if level != level:  # NaN is the only value not equal to itself
            return True
    except (TypeError, ValueError):
        pass  # pd.NA propagates instead of returning a bool; fall through to the marker check
    return str(level) in {"nan", "NaN", "<NA>", "NaT", "None"}


def _bool_key(level) -> Optional[str]:
    """Canonical "1" / "0" key for any boolean-ish level, else None."""
    tok = str(level).strip().lower()
    if tok in _BOOL_TRUE_TOKENS:
        return "1"
    if tok in _BOOL_FALSE_TOKENS:
        return "0"
    return None


def _int_key(level) -> Optional[str]:
    """"2" for 2, 2.0 and "2.0"; None when the level is not an integral number.

    NSQIP levels arrive as floats whenever a column carries NaN, so a stratum stored as
    "2" in LEVELS_CAT_PRETTY has to survive a `2.0` lookup.
    """
    try:
        f = float(str(level).strip())
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    if f != int(f):
        return None
    return str(int(f))


# Case-folded index for every level dict, built once at import. NSQIP value casing varies
# by year ("No" vs "NO", "female" vs "Female"), so lookup must be case-insensitive; a
# collision would make the fold ambiguous, so it is an import-time error.
_FOLDED_LEVELS: Dict[int, Dict[str, str]] = {}


def _folded(mapping: Dict[str, str]) -> Dict[str, str]:
    cached = _FOLDED_LEVELS.get(id(mapping))
    if cached is None:
        cached = {}
        for k, v in mapping.items():
            fk = str(k).strip().lower()
            if fk in cached and cached[fk] != v:
                raise ValueError(
                    f"case-folded level collision on {fk!r}: {cached[fk]!r} vs {v!r}. "
                    f"Two keys in the same display dict differ only by case and render "
                    f"differently, so the fold is ambiguous."
                )
            cached[fk] = v
        _FOLDED_LEVELS[id(mapping)] = cached
    return cached


def pretty_level(variable: str, level) -> str:
    """Display text for one level of a categorical or binary variable.

    Resolution order:
      1. missing (None / NaN / pd.NA)          -> MISSING_LABEL
      2. exact string key in the variable's dict
      3. case-folded key                        ("NO", "No" and "no" all land together)
      4. integral-number key                    (2.0 and "2.0" both find the "2" stratum)
      5. canonical boolean key in that dict     ("1" / "0")
      6. canonical boolean key in YESNO_PRETTY  (the fallback for any binary with no dict)
      7. de-underscored raw string

    Steps 5 and 6 are why a float column carrying NaN cannot leak `1.0` into a table stub
    or a figure legend. `has_snake_case()` cannot catch a bare `1.0`, so it is caught here
    instead. Step 6 in particular means a binary the author forgot to register still
    renders "Yes" / "No" rather than the raw number, which matters here because the
    sixteen generated POSTDISCH_ and TIMED_BOTH_ columns are all Int8 binaries.
    """
    if is_missing_level(level):
        return MISSING_LABEL

    mapping = LEVEL_PRETTY.get(variable, {})
    key = str(level).strip()

    if mapping:
        if key in mapping:
            return mapping[key]
        folded = _folded(mapping)
        fkey = key.lower()
        if fkey in folded:
            return folded[fkey]
        ikey = _int_key(level)
        if ikey is not None and ikey in mapping:
            return mapping[ikey]
        bkey = _bool_key(level)
        if bkey is not None and bkey in mapping:
            return mapping[bkey]

    bkey = _bool_key(level)
    if bkey is not None:
        return YESNO_PRETTY[bkey]

    return key.replace("_", " ")


# ---------------------------------------------------------------------------
# THE `DISCHDEST` LEVEL MAP: THE SINGLE SOURCE OF TRUTH
#
# This map must live in exactly one place, and this is that place. `02_build_cohort.py` IMPORTS it
# and does not carry its own copy; the parent project's copy inside its own
# `02_build_cohort.py` is the origin and this is a verbatim port of it, both vocabularies
# and all three level-assignment decisions.
#
# WHY A MAP AND NOT A SUBSTRING TEST. `DISCHDEST` swaps vocabulary COMPLETELY at 2021:
# `DISCHDEST = 'Home'` returns zero rows for 2021-2024, where the level is
# `Home/Permanent Residence`. Fourteen distinct levels appear across the window and they
# are not a clean taxonomy. Testing `"Home" in level` would fold `Facility Which was Home`
# into the home group, and that ACS level means the patient went to a FACILITY that had
# been their residence before surgery, which is not a discharge to the community. The
# `dischdest_category()` helper RAISES on an unmapped level and never returns a default,
# because a future PUF vintage silently folding a new level into "Home" is exactly the
# failure this map exists to prevent.
#
# THE THREE LEVEL-ASSIGNMENT DECISIONS, carried over unchanged:
#   * `Facility Which was Home`      -> Facility, NOT Home.
#   * `Against Medical Advice (AMA)` -> Other.
#   * `Expired` and `Unknown`        -> Other. A patient who expired has no discharge day
#     and is a competing event, not a same-day discharge; `02` reconciles
#     `DISCHDEST = 'Expired'` against `DOPERTOD` and the mortality flag.
#
# Downstream, `NONHOME_DISCH` is 1 for Facility, 0 for Home and MISSING for Other. It has
# a 2021 era break and carries NO trend claim: the rate steps
# from 4.98% to 3.85% as `DISCHDEST` collapses from seven facility levels to two. Report
# `DISCHDEST` by year, but make no claim that non-home discharge
# declined.
#
# This map is published as supplementary table S1; `dischdest_level_rows()` below is the
# ordered, fully renderable view `03_timing.py` writes straight to a CSV.
# ---------------------------------------------------------------------------
DISCHDEST_LEVEL_MAP: Dict[str, str] = {
    # 2012-2020 vocabulary
    "Home": "Home",
    "Rehab": "Facility",
    "Skilled Care, Not Home": "Facility",
    "Facility Which was Home": "Facility",
    "Separate Acute Care": "Facility",
    "Unskilled Facility Not Home": "Facility",
    "Hospice": "Facility",
    "Multi-level Senior Community": "Facility",
    # 2021-2024 vocabulary
    "Home/Permanent Residence": "Home",
    "Other Facility": "Facility",
    "Acute Care Hospital": "Facility",
    # both eras
    "Against Medical Advice (AMA)": "Other",
    "Expired": "Other",
    "Unknown": "Other",
}

# Which PUF vintage each level belongs to. Reported in supplement S1 so a reader can see
# that the 2021 step in `NONHOME_DISCH` is a vocabulary change and not a practice change.
DISCHDEST_ERA: Dict[str, str] = {
    "Home": "2012-2020",
    "Rehab": "2012-2020",
    "Skilled Care, Not Home": "2012-2020",
    "Facility Which was Home": "2012-2020",
    "Separate Acute Care": "2012-2020",
    "Unskilled Facility Not Home": "2012-2020",
    "Hospice": "2012-2020",
    "Multi-level Senior Community": "2012-2020",
    "Home/Permanent Residence": "2021-2024",
    "Other Facility": "2021-2024",
    "Acute Care Hospital": "2021-2024",
    "Against Medical Advice (AMA)": "2012-2024",
    "Expired": "2012-2024",
    "Unknown": "2012-2024",
}

# Publication order for supplementary table S1: grouped by collapsed category in
# DEST_CAT_ORDER order, and within a category by vocabulary era. It is a presentation
# order only; DISCHDEST_LEVEL_MAP above is the verbatim port and is the authority.
DISCHDEST_LEVEL_ORDER: Tuple[str, ...] = (
    "Home",
    "Home/Permanent Residence",
    "Rehab",
    "Skilled Care, Not Home",
    "Facility Which was Home",
    "Separate Acute Care",
    "Unskilled Facility Not Home",
    "Hospice",
    "Multi-level Senior Community",
    "Other Facility",
    "Acute Care Hospital",
    "Against Medical Advice (AMA)",
    "Expired",
    "Unknown",
)

# The marker `02_build_cohort.py` writes where `DISCHDEST` is NULL or blank. It is NOT a
# level: it never appears in DISCHDEST_LEVEL_MAP and it is what dischdest_category()
# returns for a missing destination.
DEST_CAT_MISSING = ""


def dischdest_category(level) -> str:
    """Collapse one raw `DISCHDEST` level to 'Home' / 'Facility' / 'Other'.

    RAISES ValueError on any unmapped non-blank level and NEVER returns a default. That is
    the whole point: the map exists so a PUF vintage that introduces a new destination
    string cannot have it silently folded into the home group, and a default return value
    would reintroduce exactly that failure.

    A missing destination (None, NaN, pd.NA, or the empty string) returns
    `DEST_CAT_MISSING`, the empty string, which is what the ported builder writes into
    `DEST_CAT` and what `NONHOME_DISCH` treats as missing. A blank is the absence of a
    level, not a level, so it is not an unmapped-level error.

        dischdest_category("Home")                    -> "Home"
        dischdest_category("Facility Which was Home")  -> "Facility"    NOT "Home"
        dischdest_category("Expired")                  -> "Other"
        dischdest_category(None)                       -> ""
        dischdest_category("Home Hospice")             -> ValueError
    """
    if is_missing_level(level):
        return DEST_CAT_MISSING
    key = str(level).strip()
    if not key:
        return DEST_CAT_MISSING
    try:
        return DISCHDEST_LEVEL_MAP[key]
    except KeyError:
        raise ValueError(
            f"DISCHDEST carries an unmapped level {key!r}. The PUF vocabulary swaps at "
            f"2021 and a new vintage can add another. Add it to "
            f"variable_dict.DISCHDEST_LEVEL_MAP, DISCHDEST_ERA, DISCHDEST_PRETTY and "
            f"DISCHDEST_LEVEL_ORDER with an explicit assignment, and re-run "
            f"`python3 variable_dict.py`. Do NOT match the substring 'Home': that folds "
            f"'Facility Which was Home' into the home group."
        ) from None


class DischdestLevel(NamedTuple):
    """One published row of the `DISCHDEST` level map (supplementary table S1)."""
    level: str             # the raw string, exactly as stored in the parquet
    display: str           # DISCHDEST_PRETTY
    category: str          # the collapsed value stored in DEST_CAT (ASCII)
    category_display: str  # DEST_CAT_PRETTY
    era: str               # which PUF vocabulary the level belongs to


DISCHDEST_LEVEL_TABLE: Tuple[DischdestLevel, ...] = tuple(
    DischdestLevel(
        level=_lvl,
        display=DISCHDEST_PRETTY[_lvl],
        category=DISCHDEST_LEVEL_MAP[_lvl],
        category_display=DEST_CAT_PRETTY[DISCHDEST_LEVEL_MAP[_lvl]],
        era=DISCHDEST_ERA[_lvl],
    )
    for _lvl in DISCHDEST_LEVEL_ORDER
)

# Column headers for the S1 CSV. They are themselves rendered text, so they carry no
# snake_case token; `03_timing.py` writes these keys straight out.
DISCHDEST_TABLE_COLUMNS: Tuple[str, ...] = (
    "Recorded value",
    "Discharge destination",
    "Collapsed category stored",
    "Collapsed category",
    "Years the value appears",
)


def dischdest_level_rows() -> List[Dict[str, str]]:
    """Supplementary table S1 as a list of CSV-ready dicts, in publication order.

    Every key and every value is renderable, so `03_timing.py` can hand the result to a
    CSV writer without further formatting. The raw level string is included on purpose:
    S1 exists so a reader can check the collapse rule against the recorded vocabulary.
    """
    return [
        {
            DISCHDEST_TABLE_COLUMNS[0]: r.level,
            DISCHDEST_TABLE_COLUMNS[1]: r.display,
            DISCHDEST_TABLE_COLUMNS[2]: r.category,
            DISCHDEST_TABLE_COLUMNS[3]: r.category_display,
            DISCHDEST_TABLE_COLUMNS[4]: r.era,
        }
        for r in DISCHDEST_LEVEL_TABLE
    ]


# ---------------------------------------------------------------------------
# THE `INOUT` LEVEL MAP: same pattern, same guarantee.
#
# `INOUT` is the BILLING setting and is NEVER a discharge measure or a length of stay.
# Every prior NSQIP outpatient-ACDF paper defines its exposure with it, and an earlier
# measurement found that 34,879 of 47,713 cases labelled Outpatient
# stayed at least one night. That measurement critique is one of this paper's two headline
# findings, which is why the parent project's outright ban on the column
# is lifted here and only here.
#
# Two levels, verified read-only against the curated `main` view on 2026-07-31:
# `Inpatient` (6,457,523 rows) and `Outpatient` (5,357,395 rows), plus exactly ONE SQL NULL
# row database-wide. That NULL row is not in this cohort, but the recode must not crash on
# it, so `inout_flag()` is NULL-safe and returns None rather than raising.
# ---------------------------------------------------------------------------
INOUT_LEVEL_MAP: Dict[str, int] = {
    "Inpatient": 0,
    "Outpatient": 1,
}

INOUT_OUTPATIENT_LEVEL = "Outpatient"
INOUT_INPATIENT_LEVEL = "Inpatient"


def inout_flag(level) -> Optional[int]:
    """1 for the Outpatient billing setting, 0 for Inpatient, None when missing.

    RAISES ValueError on any unmapped NON-NULL level, for the same reason
    `dischdest_category()` does: a new billing level must be assigned explicitly, not
    absorbed. NULL-safe by design, because exactly one SQL NULL row exists database-wide.
    Lookup is case-insensitive, since warehouse value casing varies by year.

    This is the derivation of `INOUT_OUTPT`. It is a BILLING flag. It is not a same-day
    discharge flag: that is `SAMEDAY_DISCH`, derived from `LOS_POSTOP` and therefore from
    `DOPTODIS`. Confusing the two is the error this paper documents.
    """
    if is_missing_level(level):
        return None
    key = str(level).strip()
    if not key:
        return None
    if key in INOUT_LEVEL_MAP:
        return INOUT_LEVEL_MAP[key]
    for k, v in INOUT_LEVEL_MAP.items():
        if k.lower() == key.lower():
            return v
    raise ValueError(
        f"INOUT carries an unmapped level {key!r}. Exactly two levels were verified "
        f"against the warehouse on 2026-07-31 ({', '.join(INOUT_LEVEL_MAP)}), plus one "
        f"SQL NULL row database-wide. Add the new level to "
        f"variable_dict.INOUT_LEVEL_MAP, INOUT_PRETTY and INOUT_ORDER with an explicit "
        f"assignment, and re-run `python3 variable_dict.py`."
    )


# ---------------------------------------------------------------------------
# CONTRACT CONSTANTS
#
# What the pre-specification fixes explicitly, exported so that 02 through 08 cannot each
# invent a parallel spelling or a different order. Anything the pre-specification does NOT
# fix is deliberately absent.
#
# TIMED_OUTCOMES is the ninth and most important of these and is defined at the top of
# this file, because the LABELS generator keys off it.
# ---------------------------------------------------------------------------

# The four components of WINDOW_COMP, the safe-discharge composite:
# hematoma requiring unplanned reoperation, unplanned reintubation, ventilator beyond
# 48 hours, and reoperative tracheostomy. Equivalently HEMATOMA OR AIRWAY_COMP, because
# the parent's three-component airway composite is fixed by the companion hematoma analysis.
#
# PNEUMONIA and READMIT_UNPLANNED are EXCLUDED and their absence is load-bearing.
# Pneumonia is a parenchymal infection, not an airway-patency event; folding it in would
# inflate the composite with a mechanistically different outcome. A readmission is by
# construction a post-discharge event and continued in-hospital observation does not
# prevent it. Both are reported as their own rows in the descriptive timing analysis.
#
# REOP_TRACH is the parent's FROZEN SLOT-1 implementation, unchanged, so the two papers
# cannot disagree about the composite. The wider
# REOP_TRACH_SLOT12 marker is for the ascertainment supplement only and is NOT here.
WINDOW_COMPONENTS: Tuple[str, ...] = ("HEMATOMA", "REINTUBATION", "VENT48", "REOP_TRACH")

# The three pre-specified residual-risk thresholds, strictly decreasing.
# Reporting several rather than one is deliberate: there is no consensus acceptable
# residual risk for discharge after ACDF, and a single threshold invites the objection
# that the number was chosen after seeing the curves. These are proportions, not
# percentages; render them with threshold_label(), never by hand.
RISK_THRESHOLDS: Tuple[float, ...] = (0.01, 0.005, 0.0025)


def threshold_label(threshold: float) -> str:
    """Display text for one pre-specified risk threshold: 0.005 -> "0.5%".

    Raises ValueError on any threshold that is not one of RISK_THRESHOLDS, so a step
    cannot quietly report a fourth threshold that was never pre-specified.
    """
    key = repr(float(threshold))
    if key in RISK_THRESHOLD_PRETTY:
        return RISK_THRESHOLD_PRETTY[key]
    raise ValueError(
        f"{threshold!r} is not one of the pre-specified RISK_THRESHOLDS "
        f"{RISK_THRESHOLDS}. The thresholds were locked before any curve was fitted; "
        f"adding one after the fact is exactly the objection the "
        f"three-threshold table exists to answer."
    )


class Stratifier(NamedTuple):
    """One locked stratifier and its ordered level strings, as stored in the parquet."""
    column: str
    levels: Tuple[str, ...]


# The four stratifiers, fixed a priori, and nothing else without a stated reason.
# No interaction strata, no data-driven cut points, no additional
# variable. `STEROID` is deliberately absent.
#
# The level strings are ASCII and are exactly what 02_build_cohort.py writes. MALE is an
# Int8 column, so its levels are given as "1" and "0"; pretty_level() folds 1, 1.0 and "1"
# onto the same display string, so a float round-trip cannot break the lookup.
STRATIFIERS: Tuple[Stratifier, ...] = (
    Stratifier("AGE_CAT", ("18-49", "50-64", "65-74", ">=75")),
    Stratifier("ASA_CAT", ("1-2", ">=3")),
    Stratifier("LEVELS_CAT", ("1", "2", ">=3")),
    Stratifier("MALE", ("1", "0")),
)

STRATIFIER_COLUMNS: Tuple[str, ...] = tuple(_s.column for _s in STRATIFIERS)

# The suppression rule, set before the curves were seen: a stratum with
# FEWER THAN this many events of the composite is suppressed rather than plotted, and the
# suppression is reported with its count. 541 hematoma events in 112,642 patients thins
# fast under stratification, and an unsuppressed curve on six events is unstable and
# over-interpretable.
MIN_STRATUM_EVENTS: int = 10

# The fixed observation window. NSQIP follow-up is 30 days from the principal operation
# and is complete by construction: censoring is administrative and non-informative, there
# is no loss to follow-up in the ordinary sense, and NO ESTIMATE EXTENDS PAST DAY 30.
# The day-30 cumulative incidence is the terminal estimate.
FOLLOWUP_DAYS: int = 30

# The pre-specified tie convention, as importable renderable text so the
# Methods sentence and the sensitivity caption cannot drift from the code. The primary
# convention is the conservative choice for a safety claim: it does not credit the
# observation window with catching an event that may in fact have occurred at home.
TIE_CONVENTION = (
    "An event recorded on the same postoperative day as discharge counts as occurring "
    "before discharge."
)
TIE_CONVENTION_SENSITIVITY = (
    "An event recorded on the same postoperative day as discharge counts as occurring "
    "after discharge."
)

# The six-term covariate set for the both-ways comparison:
# age, sex, body mass index, ASA class, fusion levels, operative year, and nothing else.
# The comparison is estimated TWICE on the same cohort with THIS identical covariate set,
# differing only in how the outpatient exposure is defined: once by `INOUT` as the prior
# literature does, once by true postoperative length of stay. The point is the definition
# change, so nothing else may vary, including the parameterisation.
#
# These are the ANALYTIC column names. The continuous terms are the per-unit-scaled
# columns the ported builder already materialises, so a coefficient is interpretable
# without rescaling. `LEVELS_CAT` and `OPERYR` enter as categorical terms;
# `05_inout.py` chooses the coding once and uses the SAME coding in both fits.
BOTHWAYS_COVARIATES: Tuple[str, ...] = (
    "AGE_PER10", "MALE", "BMI_PER5", "ASA_GE3", "LEVELS_CAT", "OPERYR",
)

# PORTED. Burden = count of these four. This project never analyses the coagulopathy
# burden (the parent paper owns it in every form), but `02_build_cohort.py`
# still materialises COAG_BURDEN_N from this roster, and dropping it would fork the two
# cohorts. Keep it; do not read it in 03 through 08.
BURDEN_COMPONENTS: Tuple[str, ...] = ("BLEEDDIS", "INR_GT12", "PLT_LT150", "PTT_GT35")

# PORTED, AND FORBIDDEN AS AN ANALYSIS ROSTER. This is the parent's consequence cascade:
# hematoma versus no hematoma rates of reintubation, ventilator beyond 48 hours,
# pneumonia, readmission, mortality and length of stay, and the ratios between them.
# Every one of those quantities is excluded from this manuscript, as is
# any length-of-stay comparison by hematoma status. It is retained ONLY so
# `verify_numbers.py` can enumerate the forbidden roster and assert that none of it
# reaches a rendered artefact. No analysis step may iterate it.
CONSEQUENCE_OUTCOMES: Tuple[str, ...] = (
    "REINTUBATION", "VENT48", "PNEUMONIA", "READMIT_UNPLANNED", "MORT30", "LOS_POSTOP",
)

# PORTED COMPLETENESS ROSTER, NOT A MODEL. The parent computed `MODEL_COMPLETE` as
# completeness on its six adjusted-model terms. This project fits no hematoma model
# and therefore exports no `PRIMARY_MODEL_TERMS`; that name is deleted on
# purpose, because a constant called "the primary model terms" invites a step to fit the
# primary model. But `MODEL_COMPLETE` is in the ported column contract
# and must be byte-identical to the parent's, or the two cohorts fork.
# This roster is the same six columns in the same order, named for what it actually does.
# READ IT ONLY to reproduce that flag. Do not fit anything on it.
MODEL_COMPLETE_TERMS: Tuple[str, ...] = (
    "BMI_PER5", "MALE", "LEVELS_PER_LEVEL", "ASA_GE3", "AGE_PER10", "HYPERMED",
)


# ---------------------------------------------------------------------------
# Section ordering for Table 1: BASELINE CHARACTERISTICS BY DISCHARGE TIMING
# (LOS_CAT 0 / 1 / >= 2 postoperative nights), which is this paper's Table 1 and is NOT
# the parent's by-hematoma-status Table 1.
#
# What changed relative to the parent, and why:
#   * The coagulopathy-burden section is GONE. It belongs to the parent in every form
#     and this paper never analyses it.
#   * The preoperative laboratory section is GONE for the same reason: the coagulation
#     panel is the parent's exposure, and a discharge-timing Table 1 gains nothing from it.
#     The labels survive because `02` still materialises the columns.
#   * A DISCHARGE AND DISPOSITION section is NEW. It carries the length of stay, both
#     discharge-timing flags, the collapsed destination and, deliberately, `INOUT`:
#     putting the billing label in the same table as the true length of stay is the first
#     view of the measurement critique the reader gets.
#   * `STEROID` is dropped from the comorbidity section (excluded from
#     every analysis and every table because its definition broadening is undated).
#   * The outcomes section adds `WINDOW_COMP`, the safe-discharge composite, and drops the
#     dysphagia columns, which belong to the parent.
#
# Every name here must exist in LABELS; that is asserted at import.
# ---------------------------------------------------------------------------
SECTIONS: Dict[str, List[str]] = {
    "Demographics": [
        "AGE", "AGE_CAT", "MALE", "RACE_CAT", "BMI",
    ],
    "Comorbidity and physiologic status": [
        "ASACLAS", "ASA_CAT", "FNSTATUS2", "DIABETES", "SMOKE", "HXCOPD", "HXCHF",
        "HYPERMED", "DIALYSIS", "BLEEDDIS",
    ],
    "Operative characteristics": [
        "LEVELS_N", "LEVELS_CAT", "ANT_INSTRUM", "OPTIME", "SPEC_CAT", "OPERYR",
    ],
    "Discharge and disposition": [
        "LOS_POSTOP", "SAMEDAY_DISCH", "DISCH_BY_POD1", "DEST_CAT", "INOUT",
    ],
    "Outcomes": [
        "HEMATOMA", "REOP_ANY", "REINTUBATION", "VENT48", "PNEUMONIA", "AIRWAY_COMP",
        "REOP_TRACH", "WINDOW_COMP", "READMIT_UNPLANNED", "MORT30",
    ],
}


# ---------------------------------------------------------------------------
# IMPORT-TIME INVARIANTS. Everything below fails at import, not at render, so a broken
# label, a drifted roster or an unmapped level cannot reach a figure or the manuscript.
# Every one of them RAISES. None of them warns.
# ---------------------------------------------------------------------------

# 1. Every variable named in SECTIONS must have a label.
for _sec, _vars in SECTIONS.items():
    for _v in _vars:
        if _v not in LABELS:
            raise ValueError(f"SECTIONS[{_sec!r}] references {_v!r}, which is not in LABELS")
del _sec, _vars, _v

# 2. Every name in every contract constant must have a label. This is what stops `06`
#    from stratifying on a column nobody labelled and `05` from adjusting for one.
for _tuple_name, _tuple in (
    ("WINDOW_COMPONENTS", WINDOW_COMPONENTS),
    ("STRATIFIER_COLUMNS", STRATIFIER_COLUMNS),
    ("BOTHWAYS_COVARIATES", BOTHWAYS_COVARIATES),
    ("BURDEN_COMPONENTS", BURDEN_COMPONENTS),
    ("CONSEQUENCE_OUTCOMES", CONSEQUENCE_OUTCOMES),
    ("MODEL_COMPLETE_TERMS", MODEL_COMPLETE_TERMS),
):
    for _v in _tuple:
        if _v not in LABELS:
            raise ValueError(f"{_tuple_name} references {_v!r}, which is not in LABELS")
del _tuple_name, _tuple, _v

# 3. TIMED_OUTCOMES: every column name the roster implies must resolve to a label, and the
#    roster itself must be structurally sound. Eight outcomes, unique keys, unique
#    suffixes, and exactly one outcome ineligible for the post-discharge fraction.
if len(TIMED_OUTCOMES) != 8:
    raise ValueError(f"TIMED_OUTCOMES must carry 8 outcomes, found {len(TIMED_OUTCOMES)}")
if len({_o.key for _o in TIMED_OUTCOMES}) != len(TIMED_OUTCOMES):
    raise ValueError("TIMED_OUTCOMES has a duplicate analysis key")
if len({_o.suffix for _o in TIMED_OUTCOMES}) != len(TIMED_OUTCOMES):
    raise ValueError("TIMED_OUTCOMES has a duplicate column suffix")
for _o in TIMED_OUTCOMES:
    for _label_key in (_o.event_col, _o.pod_col,
                       f"POSTDISCH_{_o.suffix}", f"TIMED_BOTH_{_o.suffix}") + _o.raw_fields:
        if _label_key not in LABELS:
            raise ValueError(
                f"TIMED_OUTCOMES[{_o.key!r}] references {_label_key!r}, which is not in LABELS"
            )
_ineligible = tuple(_o.key for _o in TIMED_OUTCOMES if not _o.postdisch_eligible)
if _ineligible != ("readmit",):
    raise ValueError(
        f"exactly one timed outcome, unplanned readmission, is ineligible for the "
        f"post-discharge fraction (readmission caveat); found "
        f"{_ineligible}"
    )
del _o, _label_key, _ineligible

# 4. WINDOW_COMP membership is fixed and the exclusions are load-bearing.
if len(WINDOW_COMPONENTS) != 4:
    raise ValueError(f"WINDOW_COMPONENTS must carry 4 components, found {len(WINDOW_COMPONENTS)}")
for _forbidden in ("PNEUMONIA", "OUPNEUMO", "READMIT_UNPLANNED", "UNPLANNEDREADMISSION1"):
    if _forbidden in WINDOW_COMPONENTS:
        raise ValueError(
            f"{_forbidden!r} is in WINDOW_COMPONENTS. Pneumonia and unplanned readmission "
            f"are EXCLUDED from the safe-discharge composite."
        )
del _forbidden

# 5. Every stratifier must have a registered level dict covering every declared level, or
#    a stratum label would fall through to the de-underscored fallback in a figure legend.
for _strat in STRATIFIERS:
    _levels = LEVEL_PRETTY.get(_strat.column)
    if not _levels:
        raise ValueError(
            f"STRATIFIERS names {_strat.column!r}, which has no LEVEL_PRETTY dict"
        )
    for _lv in _strat.levels:
        if _lv not in _levels:
            raise ValueError(
                f"STRATIFIERS[{_strat.column!r}] declares level {_lv!r}, which is not a "
                f"key of its LEVEL_PRETTY dict"
            )
del _strat, _levels, _lv

# 6. LABELS integrity. This runs BEFORE reconciliation on purpose: a leaking label is the
#    more fundamental defect, and checking it first means the traceback names the label
#    rather than the display dict that merely copied it.
for _k, _spec in LABELS.items():
    for _field, _txt in (("short_label", _spec.short_label), ("prose_label", _spec.prose_label)):
        if has_snake_case(_txt):
            raise ValueError(f"LABELS[{_k!r}].{_field} leaks snake_case: {_txt!r}")
        if has_em_dash(_txt):
            raise ValueError(f"LABELS[{_k!r}].{_field} contains an EM-dash: {_txt!r}")
    if _spec.var_type not in {"continuous", "binary", "categorical", "ordinal"}:
        raise ValueError(f"LABELS[{_k!r}].var_type is not a recognised type: {_spec.var_type!r}")
del _k, _spec, _field, _txt

# 7. RECONCILIATION. LABELS is the single source of display text. Every FEATURE_PRETTY key
#    must exist in LABELS and carry an IDENTICAL string, so short(k) and pretty(k) can
#    never disagree. Drift RAISES rather than being silently overwritten: a mismatch means
#    an author edited one dict and not the other, and silently picking a winner is how a
#    figure and a table end up disagreeing in print.
_drift: List[str] = []
for _k, _txt in FEATURE_PRETTY.items():
    if _k not in LABELS:
        _drift.append(f"  FEATURE_PRETTY[{_k!r}] has no LABELS entry")
    elif LABELS[_k].short_label != _txt:
        _drift.append(
            f"  FEATURE_PRETTY[{_k!r}] = {_txt!r} but LABELS[{_k!r}].short_label = "
            f"{LABELS[_k].short_label!r}"
        )
if _drift:
    raise ValueError(
        "display-dict drift against LABELS (edit LABELS, then copy the string across):\n"
        + "\n".join(_drift)
    )
del _drift, _k, _txt

# 8. Nothing user-visible may carry a snake_case leak or an EM-dash.
for _name, _d in ALL_PRETTY_DICTS.items():
    for _k, _txt in _d.items():
        if has_snake_case(_txt):
            raise ValueError(f"{_name}[{_k!r}] leaks snake_case: {_txt!r}")
        if has_em_dash(_txt):
            raise ValueError(f"{_name}[{_k!r}] contains an EM-dash: {_txt!r}")
del _name, _d, _k, _txt

# 9. Build (and therefore validate) every case-folded level index at import.
for _d in LEVEL_PRETTY.values():
    _folded(_d)
del _d

# 10. Ordered strata: the ORDER list and its display dict must agree exactly, key for key
#     and in order, or a table would print its rows in one order and its legend in another.
for _order_name, _order, _dict in (
    ("ASA_CAT_ORDER", ASA_CAT_ORDER, ASA_CAT_PRETTY),
    ("AGE_CAT_ORDER", AGE_CAT_ORDER, AGE_CAT_PRETTY),
    ("LOS_CAT_ORDER", LOS_CAT_ORDER, LOS_CAT_PRETTY),
    ("LEVELS_CAT_ORDER", LEVELS_CAT_ORDER, LEVELS_CAT_PRETTY),
    ("COAG_BURDEN_CAT_ORDER", COAG_BURDEN_CAT_ORDER, COAG_BURDEN_CAT_PRETTY),
    ("DEST_CAT_ORDER", DEST_CAT_ORDER, DEST_CAT_PRETTY),
    ("RACE_CAT_ORDER", RACE_CAT_ORDER, RACE_CAT_PRETTY),
    ("SPEC_CAT_ORDER", SPEC_CAT_ORDER, SPEC_CAT_PRETTY),
    ("INOUT_ORDER", INOUT_ORDER, INOUT_PRETTY),
):
    if list(_dict) != list(_order):
        raise ValueError(
            f"{_order_name} = {list(_order)!r} does not match its display dict key order "
            f"{list(_dict)!r}"
        )
del _order_name, _order, _dict

# 11. The `DISCHDEST` level map must cover exactly the levels the display dict knows, in
#     both directions, and every collapsed value must be a real DEST_CAT level. A level
#     present in one and absent from the other is how supplement S1 ends up incomplete.
if set(DISCHDEST_LEVEL_MAP) != set(DISCHDEST_PRETTY):
    raise ValueError(
        "DISCHDEST_LEVEL_MAP and DISCHDEST_PRETTY cover different level sets: "
        f"only in the map {sorted(set(DISCHDEST_LEVEL_MAP) - set(DISCHDEST_PRETTY))}, "
        f"only in the display dict {sorted(set(DISCHDEST_PRETTY) - set(DISCHDEST_LEVEL_MAP))}"
    )
if set(DISCHDEST_ERA) != set(DISCHDEST_LEVEL_MAP):
    raise ValueError("DISCHDEST_ERA and DISCHDEST_LEVEL_MAP cover different level sets")
if sorted(DISCHDEST_LEVEL_ORDER) != sorted(DISCHDEST_LEVEL_MAP):
    raise ValueError("DISCHDEST_LEVEL_ORDER and DISCHDEST_LEVEL_MAP cover different level sets")
if len(DISCHDEST_LEVEL_ORDER) != len(set(DISCHDEST_LEVEL_ORDER)):
    raise ValueError("DISCHDEST_LEVEL_ORDER repeats a level")
for _lvl, _cat in DISCHDEST_LEVEL_MAP.items():
    if _cat not in DEST_CAT_PRETTY:
        raise ValueError(
            f"DISCHDEST_LEVEL_MAP[{_lvl!r}] collapses to {_cat!r}, which is not a DEST_CAT level"
        )
del _lvl, _cat

# 12. THE SUBSTRING TRAP, machine-checked rather than asserted in a comment. At least one
#     level contains the word "Home" and does NOT collapse to Home, so a substring rule
#     would be provably wrong on this vocabulary. If this ever stops being true, the
#     warning in the comment above has stopped being load-bearing and should be re-read,
#     not deleted.
_substring_home = {_l for _l in DISCHDEST_LEVEL_MAP if "home" in _l.lower()}
_true_home = {_l for _l, _c in DISCHDEST_LEVEL_MAP.items() if _c == "Home"}
if _substring_home == _true_home or not (_substring_home - _true_home):
    raise ValueError(
        "the DISCHDEST substring trap no longer holds: every level containing 'Home' now "
        "collapses to Home. Re-check the level assignments before changing the map."
    )
del _substring_home, _true_home

# 13. The `INOUT` level map, same shape. Two levels, and BOTH display strings must say
#     "billing", because the entire point of the label is that a reader cannot read an
#     `INOUT` row as a length of stay.
if set(INOUT_LEVEL_MAP) != set(INOUT_PRETTY):
    raise ValueError("INOUT_LEVEL_MAP and INOUT_PRETTY cover different level sets")
if len(INOUT_LEVEL_MAP) != 2:
    raise ValueError(
        f"INOUT was verified read-only against the warehouse on 2026-07-31 to carry "
        f"exactly two levels; found {sorted(INOUT_LEVEL_MAP)}"
    )
for _lvl, _txt in INOUT_PRETTY.items():
    if "billing" not in _txt.lower():
        raise ValueError(
            f"INOUT_PRETTY[{_lvl!r}] = {_txt!r} does not say 'billing'. INOUT is the "
            f"billing label and never a discharge measure; the display string must say so."
        )
del _lvl, _txt

# 14. The pre-specified risk thresholds must be strictly decreasing and every one of them
#     must have display text, so `06_window.py` cannot print a threshold it cannot name.
if len(RISK_THRESHOLDS) != 3:
    raise ValueError(f"RISK_THRESHOLDS must carry 3 thresholds, found {RISK_THRESHOLDS}")
if list(RISK_THRESHOLDS) != sorted(RISK_THRESHOLDS, reverse=True):
    raise ValueError(f"RISK_THRESHOLDS must be strictly decreasing: {RISK_THRESHOLDS}")
for _t in RISK_THRESHOLDS:
    if repr(float(_t)) not in RISK_THRESHOLD_PRETTY:
        raise ValueError(f"RISK_THRESHOLDS carries {_t!r} with no RISK_THRESHOLD_PRETTY entry")
del _t

# 15. The pre-specification strings and the one deleted constant. TIE_CONVENTION reaches
#     the Methods verbatim, so it is held to the rendered-text bans here rather than at
#     write time. `PRIMARY_MODEL_TERMS` is deliberately absent from this module: this
#     project fits no hematoma risk model and no points score, and a
#     constant by that name would invite a step to fit one.
for _s in (TIE_CONVENTION, TIE_CONVENTION_SENSITIVITY):
    assert_renderable(_s, where="pre-specification text")
del _s
if FOLLOWUP_DAYS != 30:
    raise ValueError(f"FOLLOWUP_DAYS is fixed at 30 by the NSQIP design; found {FOLLOWUP_DAYS}")


# ---------------------------------------------------------------------------
# Self-test: `python3 variable_dict.py`. Exits 0 on success, non-zero on any failure.
# ---------------------------------------------------------------------------
def _raises(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except RenderabilityError:
        return True
    return False


def _raises_value(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _self_test() -> None:
    checks = 0

    # --- 1. the project seed -------------------------------------------------
    assert SEED == 20260731, f"SEED drifted: {SEED}"
    assert isinstance(SEED, int)
    assert SEED != 20260730, "SEED must be this project's own seed, not the parent's"
    checks += 3

    # --- 2. has_snake_case: the leak forms it exists to catch -----------------
    must_flag: Sequence[str] = [
        "los_cat",
        "(levels_n)",
        "Timing (pod_reintub)",
        "Share (postdisch_hematoma)",
        "[age_cat]",
        "los_postop,",
        "x=hematoma_pod",
        "Fusion levels, levels_cat",
        "Window (window_comp)",
    ]
    for s in must_flag:
        assert has_snake_case(s) is True, f"has_snake_case({s!r}) should be True"
        checks += 1

    must_not_flag: Sequence[str] = [
        "Discharge timing",
        "Age, years",
        "ASA class 3 or 4",
        "Body mass index, kg/m²",
        "Platelet count below 150 ×10³/µL",
        "3 or more levels",
        "2 or more nights",
        "Outpatient billing setting",
        f"18{EN_DASH}49 years",
        f"1.45{EN_DASH}4.40",
        "",
        "_",
        "__",
        MISSING_LABEL,
        TIE_CONVENTION,
        TIE_CONVENTION_SENSITIVITY,
    ]
    for s in must_not_flag:
        assert has_snake_case(s) is False, f"has_snake_case({s!r}) should be False"
        checks += 1

    # --- 3. has_em_dash and assert_renderable --------------------------------
    assert has_em_dash("a" + EM_DASH + "b") is True
    assert has_em_dash(f"1.45{EN_DASH}4.40") is False, "EN-dash must NOT be flagged"
    assert has_em_dash(f"18{EN_DASH}49 years") is False, "EN-dash must NOT be flagged"
    assert has_em_dash("Discharge timing") is False
    checks += 4

    assert _raises(assert_renderable, "Discharge" + EM_DASH + "timing"), \
        "assert_renderable must raise on an EM-dash"
    assert _raises(assert_renderable, "Share (postdisch_hematoma)"), \
        "assert_renderable must raise on snake_case leakage"
    assert _raises(assert_renderable, "Window (window_comp)")
    checks += 3

    # It returns the string unchanged when the string is clean, so it can be used inline.
    for s in [f"2.53 (1.45{EN_DASH}4.40)", pretty("LOS_POSTOP"),
              pretty_level("LOS_CAT", ">=2"), pretty_level("AGE_CAT", ">=75"),
              pretty_level("INOUT", "Outpatient")]:
        assert assert_renderable(s) == s
        checks += 1

    # --- 4. no label and no display string anywhere leaks ---------------------
    for dname, d in ALL_PRETTY_DICTS.items():
        for k, txt in d.items():
            assert not has_snake_case(txt), f"{dname}[{k!r}] leaks snake_case: {txt!r}"
            assert not has_em_dash(txt), f"{dname}[{k!r}] contains an EM-dash: {txt!r}"
            checks += 2
    for k, spec in LABELS.items():
        assert not has_snake_case(spec.short_label), f"LABELS[{k!r}].short_label leaks: {spec.short_label!r}"
        assert not has_snake_case(spec.prose_label), f"LABELS[{k!r}].prose_label leaks: {spec.prose_label!r}"
        assert not has_em_dash(spec.short_label) and not has_em_dash(spec.prose_label), \
            f"LABELS[{k!r}] contains an EM-dash"
        checks += 3

    # --- 5. pretty() / short() invariant for every LABELS key -----------------
    for k in LABELS:
        assert pretty(k) == short(k) == LABELS[k].short_label, \
            f"pretty({k!r})={pretty(k)!r} != short({k!r})={short(k)!r}"
        checks += 1
    shared = sorted(set(LABELS) & set(FEATURE_PRETTY))
    assert len(shared) == len(FEATURE_PRETTY), \
        "every FEATURE_PRETTY key must exist in LABELS (import-time reconciliation)"
    for k in shared:
        assert FEATURE_PRETTY[k] == LABELS[k].short_label == pretty(k), f"display drift on {k!r}"
        checks += 1
    # The fallback degrades to readable text rather than leaking.
    assert not has_snake_case(pretty("a_key_that_does_not_exist"))
    assert pretty("a_key_that_does_not_exist") == "A key that does not exist"
    checks += 2

    # accessors round-trip
    assert unit("AGE") == "years" and unit("LOS_POSTOP") == "days" and unit("INOUT") == ""
    assert var_type("SAMEDAY_DISCH") == "binary" and var_type("LOS_CAT") == "categorical"
    assert prose("LOS_POSTOP") == "postoperative length of stay"
    checks += 3

    # The deleted keys must STAY deleted. This project fits no score, and a label for a
    # column that can never exist invites a step to reach for it.
    for gone in ("SCORE_TOTAL", "SCORE_CAT", "SCORE_WITH_COAG"):
        assert gone not in LABELS, f"{gone!r} was deleted from this project and came back"
        checks += 1
    assert "PRIMARY_MODEL_TERMS" not in globals(), \
        "PRIMARY_MODEL_TERMS was deleted: this project fits no hematoma model"
    checks += 1

    # --- 6. pretty_level(): every equivalent spelling folds together ----------
    true_forms = [1, 1.0, True, "1", "1.0", "True", "true", "TRUE", "Yes", "yes", "YES", " 1 "]
    false_forms = [0, 0.0, False, "0", "0.0", "False", "false", "FALSE", "No", "no", "NO", " 0 "]
    missing_forms = [None, float("nan")]

    # (a) a variable with a dedicated dict
    for v in true_forms:
        got = pretty_level("MALE", v)
        assert got == "Male", f"pretty_level('MALE', {v!r}) -> {got!r}"
        checks += 1
    for v in false_forms:
        got = pretty_level("MALE", v)
        assert got == "Female", f"pretty_level('MALE', {v!r}) -> {got!r}"
        checks += 1

    # (b) a binary with NO dedicated dict must still fold to Yes / No, never to "1.0".
    #     The sixteen generated POSTDISCH_ and TIMED_BOTH_ columns are exactly this case.
    for col in ("HEMATOMA", "SAMEDAY_DISCH", "DISCH_BY_POD1", "WINDOW_COMP",
                "POSTDISCH_HEMATOMA", "TIMED_BOTH_AIRWAY", "INOUT_OUTPT",
                "LOS_POSTOP_AGREE", "ERA_ICD10", "REOP_TRACH_SLOT12"):
        for v in true_forms:
            assert pretty_level(col, v) == "Yes", f"pretty_level({col!r}, {v!r})"
            checks += 1
        for v in false_forms:
            assert pretty_level(col, v) == "No", f"pretty_level({col!r}, {v!r})"
            checks += 1
    for v in true_forms + false_forms:
        got = pretty_level("A_COLUMN_NOBODY_REGISTERED", v)
        assert got in {"Yes", "No"}, f"unregistered binary leaked {got!r}"
        checks += 1

    # (c) missing folds onto one label everywhere
    for var in ["MALE", "HEMATOMA", "LEVELS_CAT", "AGE_CAT", "LOS_CAT", "INOUT",
                "DIABETES", "A_MYSTERY_COLUMN"]:
        for v in missing_forms:
            assert pretty_level(var, v) == MISSING_LABEL, \
                f"pretty_level({var!r}, {v!r}) should be {MISSING_LABEL!r}"
            checks += 1

    # (d) NOTHING boolean-ish or missing may ever render as a bare number or "nan"
    forbidden = {"1", "0", "1.0", "0.0", "True", "False", "nan", "NaN", "None", "<NA>"}
    for var in ["MALE", "HEMATOMA", "VENT48", "WINDOW_COMP", "POSTDISCH_READMIT",
                "A_MYSTERY_COLUMN"]:
        for v in true_forms + false_forms + missing_forms:
            out = pretty_level(var, v)
            assert out not in forbidden, f"pretty_level({var!r}, {v!r}) leaked {out!r}"
            assert not has_snake_case(out) and not has_em_dash(out)
            checks += 1

    # (e) NSQIP casing varies by year: DIABETES is "NO", BLEEDDIS is "No"
    assert pretty_level("DIABETES", "NO") == "None"
    assert pretty_level("DIABETES", "No") == "None"
    assert pretty_level("DIABETES", "no") == "None"
    assert pretty_level("DIABETES", "NON-INSULIN") == "Non-insulin dependent"
    assert pretty_level("DIABETES", "Non-Insulin") == "Non-insulin dependent"
    assert pretty_level("DIABETES", "INSULIN") == "Insulin dependent"
    checks += 6

    assert pretty_level("SEX", "female") == "Female"
    assert pretty_level("SEX", "Female") == "Female"
    assert pretty_level("SEX", "FEMALE") == "Female"
    assert pretty_level("SEX", "male") == "Male"
    checks += 4

    assert pretty_level("REINTUB", "No Complication") == "No"
    assert pretty_level("REINTUB", "NO COMPLICATION") == "No"
    assert pretty_level("REINTUB", "Unplanned Intubation") == "Yes"
    assert pretty_level("FAILWEAN", "On Ventilator greater than 48 Hours") == "Yes"
    assert pretty_level("OUPNEUMO", "Pneumonia") == "Yes"
    checks += 5

    # (f) ordered strata: ASCII keys in, prose out; floats find their stratum
    assert list(LEVELS_CAT_PRETTY) == LEVELS_CAT_ORDER == ["1", "2", ">=3"]
    assert pretty_level("LEVELS_CAT", "1") == "1 level"
    assert pretty_level("LEVELS_CAT", 2) == "2 levels"
    assert pretty_level("LEVELS_CAT", 2.0) == "2 levels"
    assert pretty_level("LEVELS_CAT", "2.0") == "2 levels"
    assert pretty_level("LEVELS_CAT", ">=3") == "3 or more levels"
    checks += 6

    assert list(COAG_BURDEN_CAT_PRETTY) == COAG_BURDEN_CAT_ORDER == ["0", "1", ">=2"]
    assert pretty_level("COAG_BURDEN_CAT", 0) == "No abnormality"
    assert pretty_level("COAG_BURDEN_CAT", ">=2") == "Two or more abnormalities"
    checks += 3

    assert pretty_level("ASACLAS", "3-Severe Disturb") == "Class 3"
    assert pretty_level("ASA_CAT", ">=3") == "Class 3 or 4"
    assert pretty_level("SPEC_CAT", "Neurosurgery") == "Neurosurgery"
    assert pretty_level("DEST_CAT", "Facility") == "Rehabilitation or skilled nursing facility"
    assert pretty_level("DISCHDEST", "Skilled Care, Not Home") == "Skilled nursing facility"
    assert pretty_level("RACE_CAT", "Black") == "Black or African American"
    checks += 6

    # (g) a string level is NEVER missing: PRSEPIS "None" is a real category
    assert is_missing_level("None") is False
    assert is_missing_level("nan") is False
    assert is_missing_level(None) is True
    assert is_missing_level(float("nan")) is True
    assert pretty_level("PRSEPIS", "None") == "None"
    assert pretty_level("PRSEPIS", "SIRS") == "Systemic inflammatory response syndrome"
    checks += 6

    # (h) every registered level renders to its declared string and never leaks
    for var, mapping in LEVEL_PRETTY.items():
        for lvl in mapping:
            out = pretty_level(var, lvl)
            assert out == mapping[lvl], f"pretty_level({var!r}, {lvl!r}) -> {out!r}"
            assert not has_snake_case(out) and not has_em_dash(out)
            checks += 2

    # --- 7. formatting helpers on worked examples -----------------------------
    assert fmt_count(112642) == "112,642"
    assert fmt_count(541) == "541"
    checks += 2

    assert fmt_pct(0.477, 3) == "0.477%"
    assert fmt_pct(17.8) == "17.8%"
    assert fmt_pct(41.3) == "41.3%"
    checks += 3

    assert fmt_n_pct(541, 112642) == "541 (0.5%)"
    assert fmt_n_pct(541, 112642, 3) == "541 (0.480%)"
    assert fmt_n_pct(0, 0) == "0 (0.0%)"
    assert fmt_n_pct_denom(541, 112642, 3) == "541/112,642 (0.480%)"
    assert fmt_n_pct_denom(34879, 47713) == "34,879/47,713 (73.1%)"
    checks += 5

    assert fmt_mean_sd(54.2, 11.8) == "54.2 (11.8)"
    assert fmt_mean_sd(29.61, 6.204, 2) == "29.61 (6.20)"
    checks += 2

    assert fmt_iqr(2, 1, 4, 0) == f"2 (1{EN_DASH}4)"
    assert fmt_iqr(1.0, 1.0, 2.0) == f"1.0 (1.0{EN_DASH}2.0)"
    assert fmt_median_iqr is fmt_iqr
    checks += 3

    assert fmt_or_ci(2.53, 1.45, 4.40) == f"2.53 (1.45{EN_DASH}4.40)"
    assert fmt_rr_ci(1.97, 1.14, 3.40) == f"1.97 (1.14{EN_DASH}3.40)"
    assert fmt_ci(1.08, 1.74) == f"1.08{EN_DASH}1.74"
    checks += 3

    assert fmt_beta_se(0.928, 0.283) == "0.928 (0.283)"
    assert fmt_smd(-0.0421) == "0.042"
    checks += 2

    assert fmt_p(0.371) == "0.371"
    assert fmt_p(0.001) == "0.001"
    assert fmt_p(0.0009) == "< 0.001"
    assert fmt_p(0.0) == "< 0.001"
    assert fmt_p(0.0004, spaced=False) == "<0.001"
    for p in [0.0, 1e-12, 1e-6, 0.0004, 0.0009999, 0.001, 0.009, 0.371, 1.0]:
        assert fmt_p(p) != "0.000", f"fmt_p({p}) rendered 0.000"
        checks += 1
    checks += 5

    # every formatter output must itself be renderable
    for s in [
        fmt_count(112642), fmt_pct(0.477, 3), fmt_n_pct(541, 112642),
        fmt_n_pct_denom(541, 112642), fmt_mean_sd(54.2, 11.8), fmt_iqr(2, 1, 4, 0),
        fmt_or_ci(2.53, 1.45, 4.40), fmt_rr_ci(1.97, 1.14, 3.40), fmt_ci(1.08, 1.74),
        fmt_beta_se(0.928, 0.283), fmt_smd(-0.0421), fmt_p(0.0009), fmt_p(0.371),
    ]:
        assert assert_renderable(s) == s
        checks += 1
    for s in [fmt_iqr(2, 1, 4, 0), fmt_or_ci(2.53, 1.45, 4.40), fmt_ci(1.08, 1.74)]:
        assert EN_DASH in s and EM_DASH not in s, f"wrong range separator in {s!r}"
        checks += 1

    # --- 8. contract constants all resolve ------------------------------------
    for name in (WINDOW_COMPONENTS + STRATIFIER_COLUMNS + BOTHWAYS_COVARIATES
                 + BURDEN_COMPONENTS + CONSEQUENCE_OUTCOMES + MODEL_COMPLETE_TERMS):
        assert name in LABELS
        assert assert_renderable(pretty(name)) == pretty(name)
        checks += 2

    assert WINDOW_COMPONENTS == ("HEMATOMA", "REINTUBATION", "VENT48", "REOP_TRACH")
    assert "PNEUMONIA" not in WINDOW_COMPONENTS and "READMIT_UNPLANNED" not in WINDOW_COMPONENTS
    assert len(BOTHWAYS_COVARIATES) == 6 and len(set(BOTHWAYS_COVARIATES)) == 6
    assert len(BURDEN_COMPONENTS) == 4
    assert RISK_THRESHOLDS == (0.01, 0.005, 0.0025)
    assert MIN_STRATUM_EVENTS == 10
    assert FOLLOWUP_DAYS == 30
    checks += 7

    assert threshold_label(0.01) == "1%"
    assert threshold_label(0.005) == "0.5%"
    assert threshold_label(0.0025) == "0.25%"
    assert _raises_value(threshold_label, 0.02), \
        "threshold_label must raise on a threshold that was never pre-specified"
    checks += 4

    assert len(STRATIFIERS) == 4
    assert STRATIFIER_COLUMNS == ("AGE_CAT", "ASA_CAT", "LEVELS_CAT", "MALE")
    assert "STEROID" not in STRATIFIER_COLUMNS and "STEROID" not in BOTHWAYS_COVARIATES
    for strat in STRATIFIERS:
        for lvl in strat.levels:
            out = pretty_level(strat.column, lvl)
            assert assert_renderable(out) == out
            assert out != lvl, f"stratum {strat.column}={lvl!r} rendered as its raw level"
            checks += 2
    checks += 3

    # TIE_CONVENTION and its sensitivity are importable rather than retyped, and both are
    # fit to be pasted straight into the Methods.
    assert TIE_CONVENTION != TIE_CONVENTION_SENSITIVITY
    assert assert_renderable(TIE_CONVENTION) == TIE_CONVENTION
    assert assert_renderable(TIE_CONVENTION_SENSITIVITY) == TIE_CONVENTION_SENSITIVITY
    checks += 3

    # --- 9. SECTIONS renders end to end ---------------------------------------
    for sec, names in SECTIONS.items():
        assert assert_renderable(sec) == sec
        for n in names:
            assert n in LABELS
            assert assert_renderable(short(n)) == short(n)
            checks += 2
        checks += 1
    assert "Discharge and disposition" in SECTIONS
    assert "INOUT" in SECTIONS["Discharge and disposition"]
    assert "LOS_POSTOP" in SECTIONS["Discharge and disposition"]
    assert "Coagulopathy burden" not in SECTIONS, \
        "the coagulopathy-burden section belongs to the parent"
    assert all("STEROID" not in v for v in SECTIONS.values()), \
        "STEROID is excluded from every table"
    checks += 5

    # --- 10. THE TIMED-OUTCOME ROSTER -----------------------------------------
    assert len(TIMED_OUTCOMES) == 8
    assert TIMED_OUTCOME_KEYS == (
        "hematoma", "reop_any", "reintub", "vent48", "pneumonia", "readmit", "death", "airway"
    )
    checks += 2

    for o in TIMED_OUTCOMES:
        # every column the roster implies resolves to a label and renders cleanly
        for col in (o.event_col, o.pod_col, postdisch_column(o.key), timed_both_column(o.key)):
            assert col in LABELS, f"{col!r} (from {o.key!r}) is not in LABELS"
            assert assert_renderable(short(col)) == short(col)
            checks += 2
        for raw in o.raw_fields:
            assert raw in LABELS, f"{raw!r} (raw timing field of {o.key!r}) is not in LABELS"
            checks += 1
        # the accessors and the generated tuples agree
        assert timed_outcome(o.key) is o
        assert pod_column(o.key) == o.pod_col
        assert postdisch_column(o.key) == f"POSTDISCH_{o.suffix}"
        assert timed_both_column(o.key) == f"TIMED_BOTH_{o.suffix}"
        checks += 4
        # the generated labels are built from the one noun, so the three families cannot
        # describe the same outcome three different ways
        assert short(o.pod_col) == f"Postoperative day of {o.noun}"
        assert o.noun in short(postdisch_column(o.key)).lower()
        assert o.noun in short(timed_both_column(o.key)).lower()
        checks += 3

    assert POD_COLUMNS == tuple(o.pod_col for o in TIMED_OUTCOMES)
    assert len(set(POD_COLUMNS)) == 8
    assert len(set(POSTDISCH_COLUMNS)) == 8
    assert len(set(TIMED_BOTH_COLUMNS)) == 8
    checks += 4

    # hematoma reuses the parent's HEMATOMA_POD; there is no second POD_HEMATOMA column
    assert pod_column("hematoma") == "HEMATOMA_POD"
    assert "POD_HEMATOMA" not in LABELS, \
        "the roster points hematoma at HEMATOMA_POD; a second column would fork the day field"
    assert short("HEMATOMA_POD") == "Postoperative day of hematoma reoperation"
    checks += 3

    # unplanned readmission is the ONE outcome whose post-discharge fraction is
    # uninformative by construction (readmission caveat)
    assert timed_outcome("readmit").postdisch_eligible is False
    assert all(o.postdisch_eligible for o in TIMED_OUTCOMES if o.key != "readmit")
    assert timed_outcome("readmit").raw_fields == ("READMPODAYS1",)
    checks += 3

    # the airway composite is timed as the earliest of its three component day fields
    assert timed_outcome("airway").raw_fields == ("DREINTUB", "DFAILWEAN", "RETORPODAYS")
    assert timed_outcome("airway").event_col == "AIRWAY_COMP"
    checks += 2

    try:
        timed_outcome("not_an_outcome")
        raise AssertionError("timed_outcome must raise on an unknown key")
    except KeyError:
        pass
    checks += 1

    # --- 11. THE DISCHDEST LEVEL MAP ------------------------------------------
    assert len(DISCHDEST_LEVEL_MAP) == 14
    assert set(DISCHDEST_LEVEL_MAP) == set(DISCHDEST_PRETTY) == set(DISCHDEST_ERA)
    checks += 2

    # the three carried-over level assignments
    assert dischdest_category("Facility Which was Home") == "Facility", \
        "'Facility Which was Home' is a FACILITY, not a home discharge"
    assert dischdest_category("Facility Which was Home") != "Home"
    assert dischdest_category("Against Medical Advice (AMA)") == "Other"
    assert dischdest_category("Expired") == "Other"
    assert dischdest_category("Unknown") == "Other"
    checks += 5

    # both vocabularies collapse to the same home category
    assert dischdest_category("Home") == "Home"
    assert dischdest_category("Home/Permanent Residence") == "Home"
    assert dischdest_category("Rehab") == "Facility"
    assert dischdest_category("Acute Care Hospital") == "Facility"
    assert dischdest_category("Other Facility") == "Facility"
    checks += 5

    # SUBSTRING MATCHING "Home" WOULD BE WRONG, and the map is what prevents it.
    substring_home = sorted(l for l in DISCHDEST_LEVEL_MAP if "home" in l.lower())
    true_home = sorted(l for l, c in DISCHDEST_LEVEL_MAP.items() if c == "Home")
    assert substring_home != true_home, \
        "if these ever agree, the substring shortcut would be safe and this test is stale"
    assert "Facility Which was Home" in substring_home
    assert "Facility Which was Home" not in true_home
    assert "Skilled Care, Not Home" in substring_home
    assert "Skilled Care, Not Home" not in true_home
    assert "Unskilled Facility Not Home" in substring_home
    assert "Unskilled Facility Not Home" not in true_home
    assert true_home == ["Home", "Home/Permanent Residence"]
    checks += 7

    # an unmapped level RAISES and never returns a default
    for bad in ["Home Hospice", "Long Term Care", "home", "HOME", "Skilled Nursing Facility", "x"]:
        assert _raises_value(dischdest_category, bad), \
            f"dischdest_category({bad!r}) must raise rather than default"
        checks += 1

    # a blank destination is the absence of a level, not an unmapped level
    for blank in [None, float("nan"), "", "   "]:
        assert dischdest_category(blank) == DEST_CAT_MISSING
        checks += 1

    # the published S1 view is ordered, complete and fully renderable
    rows = dischdest_level_rows()
    assert len(rows) == 14 == len(DISCHDEST_LEVEL_TABLE)
    assert [r["Recorded value"] for r in rows] == list(DISCHDEST_LEVEL_ORDER)
    for r in rows:
        assert set(r) == set(DISCHDEST_TABLE_COLUMNS)
        for col, val in r.items():
            assert assert_renderable(col) == col
            assert assert_renderable(val) == val
        assert r["Collapsed category stored"] == dischdest_category(r["Recorded value"])
        assert r["Collapsed category"] == DEST_CAT_PRETTY[r["Collapsed category stored"]]
        assert r["Discharge destination"] == DISCHDEST_PRETTY[r["Recorded value"]]
        checks += 4
    assert [r["Collapsed category stored"] for r in rows[:2]] == ["Home", "Home"]
    checks += 3

    # --- 12. THE INOUT LEVEL MAP ----------------------------------------------
    assert set(INOUT_LEVEL_MAP) == {"Inpatient", "Outpatient"}
    assert inout_flag("Outpatient") == 1
    assert inout_flag("Inpatient") == 0
    assert inout_flag("outpatient") == 1        # casing varies by year
    assert inout_flag(" Inpatient ") == 0
    checks += 5

    # NULL-safe: exactly one SQL NULL row exists database-wide and must not crash
    for null_form in [None, float("nan"), "", "   "]:
        assert inout_flag(null_form) is None, f"inout_flag({null_form!r}) must be None"
        checks += 1

    # an unmapped non-null level RAISES
    for bad in ["Observation", "Ambulatory", "Same Day", "23 Hour Observation", "Unknown"]:
        assert _raises_value(inout_flag, bad), f"inout_flag({bad!r}) must raise"
        checks += 1

    # the display labels say "billing", so an INOUT row cannot be read as a length of stay
    for lvl, txt in INOUT_PRETTY.items():
        assert "billing" in txt.lower(), f"INOUT_PRETTY[{lvl!r}] does not say billing"
        assert assert_renderable(txt) == txt
        checks += 2
    assert "billing" in short("INOUT").lower()
    assert "billing" in short("INOUT_OUTPT").lower()
    assert "billing" in prose("INOUT").lower()
    checks += 3

    # --- 13. every NEW level renders cleanly ----------------------------------
    assert list(AGE_CAT_PRETTY) == AGE_CAT_ORDER == ["18-49", "50-64", "65-74", ">=75"]
    assert pretty_level("AGE_CAT", "18-49") == f"18{EN_DASH}49 years"
    assert pretty_level("AGE_CAT", ">=75") == "75 years or older"
    checks += 3

    assert list(LOS_CAT_PRETTY) == LOS_CAT_ORDER == ["0", "1", ">=2"]
    assert pretty_level("LOS_CAT", "0") == "Same day"
    assert pretty_level("LOS_CAT", 0) == "Same day"
    assert pretty_level("LOS_CAT", 0.0) == "Same day"
    assert pretty_level("LOS_CAT", "1") == "1 night"
    assert pretty_level("LOS_CAT", 1.0) == "1 night"
    assert pretty_level("LOS_CAT", ">=2") == "2 or more nights"
    checks += 7

    assert pretty_level("INOUT", "Inpatient") == "Inpatient billing setting"
    assert pretty_level("INOUT", "Outpatient") == "Outpatient billing setting"
    checks += 2

    for var in ("AGE_CAT", "LOS_CAT", "INOUT", "DEST_CAT", "DISCHDEST"):
        for lvl in LEVEL_PRETTY[var]:
            out = pretty_level(var, lvl)
            assert assert_renderable(out) == out
            assert out not in {"nan", "None", "1.0", "0.0"}
            checks += 2

    print(f"variable_dict self-test: {checks} assertions passed")
    print(f"  SEED .............................. {SEED}")
    print(f"  LABELS entries .................... {len(LABELS)}")
    print(f"  FEATURE_PRETTY entries ............ {len(FEATURE_PRETTY)} "
          f"(all {len(shared)} reconciled against LABELS)")
    print(f"  display dicts swept ............... {len(ALL_PRETTY_DICTS)}")
    print(f"  level dicts registered ............ {len(LEVEL_PRETTY)}")
    print(f"  Table 1 sections .................. {len(SECTIONS)}")
    print(f"  timed outcomes .................... {len(TIMED_OUTCOMES)} "
          f"({', '.join(TIMED_OUTCOME_KEYS)})")
    print(f"  generated timing columns .......... {len(POD_COLUMNS) + len(POSTDISCH_COLUMNS) + len(TIMED_BOTH_COLUMNS)} "
          f"(3 families x {len(TIMED_OUTCOMES)} outcomes)")
    print(f"  post-discharge fraction reported .. {sum(o.postdisch_eligible for o in TIMED_OUTCOMES)} "
          f"of {len(TIMED_OUTCOMES)} (readmission is shape only)")
    print(f"  window composite components ....... {len(WINDOW_COMPONENTS)} "
          f"({', '.join(WINDOW_COMPONENTS)})")
    print(f"  stratifiers ....................... {len(STRATIFIERS)} "
          f"({', '.join(STRATIFIER_COLUMNS)}), suppress below {MIN_STRATUM_EVENTS} events")
    print(f"  risk thresholds ................... {', '.join(threshold_label(t) for t in RISK_THRESHOLDS)}")
    print(f"  DISCHDEST levels mapped ........... {len(DISCHDEST_LEVEL_MAP)} "
          f"(Home {sum(1 for c in DISCHDEST_LEVEL_MAP.values() if c == 'Home')}, "
          f"Facility {sum(1 for c in DISCHDEST_LEVEL_MAP.values() if c == 'Facility')}, "
          f"Other {sum(1 for c in DISCHDEST_LEVEL_MAP.values() if c == 'Other')})")
    print(f"  INOUT levels mapped ............... {len(INOUT_LEVEL_MAP)} "
          f"({', '.join(INOUT_LEVEL_MAP)}), NULL-safe")
    print(f"  follow-up window .................. {FOLLOWUP_DAYS} days, no extrapolation")
    print(f"  tie convention .................... {TIE_CONVENTION}")
    print(f"  missing-level label ............... {MISSING_LABEL!r}")
    print(f"  range separator ................... EN-dash U+2013 "
          f"(example: {fmt_or_ci(2.53, 1.45, 4.40)})")


if __name__ == "__main__":
    _self_test()
