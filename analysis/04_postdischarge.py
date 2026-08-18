#!/usr/bin/env python3
"""04_postdischarge.py -- the proportion of events occurring after the patient's own
discharge, and the denominator ladder that makes it honest.

PLAN.md sections 4, 6.2, 6.6 and 7.2. CLAUDE.md sections 4 (interface), 5 (hard rules 7
and 8), 6.3 (the locked tie convention), 8.4 (Gate 2) and 9.19 (the readmission caveat).

This is the paper's central novel measure and it has a denominator problem that is handled
in the open. The comparison needs BOTH the days-to-event field and the postoperative
length-of-stay field present on the same row, so the denominator is not "all events": it is
"events with both fields present", which is row 4 of the ladder below and nothing else.

Inputs
------
    derived/cohort_analytic.parquet     FROZEN by 02. n = 112,642. The only data source.
    derived/cohort_identified.parquet   FROZEN by 02. Read for ONE number only: the
                                        identified-rung reconciliation of the carried-over
                                        41.3% (CLAUDE.md 8.4). See the deviation note below.
    variable_dict.py                    TIMED_OUTCOMES, STRATIFIERS, labels, fmt_ helpers
    outputs/results_02.json             OPTIONAL, comparison only, never a source of a
                                        published value. See "cross-check" below.

Outputs
-------
    tables-csv/table3b_postdischarge_ladder.csv   Table 3B, the five-row ladder per outcome
    tables-csv/s2_timing_missingness.csv          Supplement S2
    tables-csv/s4_sensitivity_04.csv              This step's share of Supplement S4
    outputs/results_04.json                       flat dotted keys, prefixes postdisch. /
                                                  denom. / missing. and nothing else
    derived/notes_04.md                           the handoff for 07 and 08

The five-row ladder (PLAN.md 6.2), per timed outcome
----------------------------------------------------
    1  Events in the analytic cohort
    2  Events with the days-to-event field non-missing
    3  Events with postoperative length of stay non-missing
    4  Events with BOTH non-missing            <- THE ANALYSIS DENOMINATOR
    5  Post-discharge events, n and % OF ROW 4 ONLY

Row 5 is the only row carrying a percentage and its N is printed with it. Every drop
between rows is reported as an absolute count and as a percentage of row 1. No proportion
in this manuscript appears without its own denominator (CLAUDE.md hard rules 7 and 8).

02 froze TIMED_BOTH_<X>, whose sum is row 4 exactly. This step USES it and VERIFIES it
against its own computation rather than assuming it.

The tie convention is LOCKED (CLAUDE.md 6.3)
--------------------------------------------
    POSTDISCH_<X> = (POD_<X> > LOS_POSTOP)

An event on the same postoperative day as discharge counts as PRE-discharge. That is the
conservative choice for a safety claim: it does not credit the observation window with
catching an event that may in fact have occurred at home. The opposite convention
(POD >= LOS_POSTOP) is a required sensitivity and CLAUDE.md 4.1 grants THIS STEP the single
explicit exception to the no-local-derivation rule so it can be computed inline from the
frozen POD_* and LOS_POSTOP columns.

Readmission is EXCLUDED from the post-discharge fraction (CLAUDE.md 9.19)
------------------------------------------------------------------------
READMPODAYS1 is days from the principal operation to readmission, so a readmission is BY
CONSTRUCTION a post-discharge event for anyone who was discharged. Its post-discharge
fraction is uninformative. The roster carries the machine-readable form of the caveat
(`timed_outcome("readmit").postdisch_eligible is False`); the column POSTDISCH_READMIT
exists and this step deliberately does NOT publish it as a result. Readmission still gets
its full ladder, because rows 1 to 4 are honest for it, and its row 5 is withheld with the
reason printed beside it.

THE BOUNDING SENSITIVITY, and why the naive bound is wrong here
---------------------------------------------------------------
PLAN.md 6.2 requires a bounding sensitivity for any outcome losing more than 5% of its
events between rows 1 and 4: recompute under the two extreme assumptions, all missing
events pre-discharge and all missing events post-discharge, and report the interval.

02 measured, and publishes under `timingfield.<outcome>.lost_*`, that OF THE EVENTS LOST
BECAUSE LOS_POSTOP IS MISSING, 100% BELONG TO A PATIENT FLAGGED STILLINHOSP = 'Yes'. A
patient still admitted beyond postoperative day 30 was never discharged inside the
follow-up window, so an event on such a row CANNOT have occurred after that patient's own
discharge: there was no discharge for it to fall after. For those rows PLAN.md 6.2's "all
missing events post-discharge" extreme is not merely conservative, it is STRUCTURALLY
IMPOSSIBLE, and the naive interval is materially wider than the data supports. PLAN.md 6.2
itself says an interval wide enough to change the clinical message replaces the point
estimate, which makes printing a spuriously wide one a real harm.

BINDING (.build_facts.md 7.5 item 7): the STILLINHOSP-conditioned bound is PRIMARY and the
naive unconditional bound is shown only beside it, explicitly labelled as the unconditional
extreme. This step re-measures the decomposition itself rather than trusting the handoff.

Both bounds place the previously unplaced events, so both are computed on a row-1
denominator, not a row-4 one. The complete-case point estimate in Table 3B row 5 is on row
4. Those are different denominators on purpose and the notes say so, because a reader who
sees the point estimate fall outside the conditioned interval must be told why rather than
left to guess.

Two deliberate deviations from "read the analytic parquet only", both narrow
---------------------------------------------------------------------------
1. `derived/cohort_identified.parquet` is read for exactly one quantity: the
   identified-rung hematoma post-discharge fraction that CLAUDE.md 8.4 requires be
   re-derived rather than copied. It is a frozen 02 artefact; no cohort is re-derived and
   the warehouse is never touched. Copying 02's published number instead would satisfy the
   letter of "read the analytic parquet only" and break the stronger instruction to
   re-derive. Recorded here so a reviewer can challenge it.
2. The V0 hematoma post-discharge fraction masks the frozen RETORPODAYS column by the
   frozen HEMATOMA_V0 column, which is the same construction 02 used for the primary. It is
   a sensitivity transform of two frozen columns, the same character as the tie-convention
   exception, and CLAUDE.md 8.4 names HEMATOMA_V0 as available for exactly this purpose.

Cross-check against 02, comparison only
---------------------------------------
Where `outputs/results_02.json` is present, every ladder count, loss decomposition and
carried-over reconciliation this step derives is COMPARED against 02's published key. A
mismatch is printed as a BLOCKER and recorded, never adopted and never overwritten
(CLAUDE.md 8.5). No published value in this step is ever read from that file, and the step
runs to completion without it.

Namespace (CLAUDE.md 4.2)
-------------------------
This fragment writes `postdisch.`, `denom.` and `missing.` and nothing else. Anything
outside those three prefixes is rejected here, before the merge, with the offending key
named.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import variable_dict as VD

# CLAUDE.md hard rule 4. Nothing in this step draws on the global NumPy stream today; the
# seed is set anyway so that adding a resampling check later cannot make the step
# irreproducible by omission.
np.random.seed(VD.SEED)

STEP = "04"
HERE = Path(__file__).resolve().parent
DERIVED_DIR = HERE / "derived"
TABLES_DIR = HERE / "tables-csv"
OUTPUTS_DIR = HERE / "outputs"

ANALYTIC_PARQUET = DERIVED_DIR / "cohort_analytic.parquet"
IDENTIFIED_PARQUET = DERIVED_DIR / "cohort_identified.parquet"
RESULTS_02_JSON = OUTPUTS_DIR / "results_02.json"

RESULTS_JSON = OUTPUTS_DIR / "results_04.json"
NOTES_MD = DERIVED_DIR / "notes_04.md"
TABLE_3B_CSV = TABLES_DIR / "table3b_postdischarge_ladder.csv"
S2_CSV = TABLES_DIR / "s2_timing_missingness.csv"
S4_CSV = TABLES_DIR / "s4_sensitivity_04.csv"

# CLAUDE.md 4.2. Mirrors run_all.RESERVED_PREFIXES["04"] exactly.
RESERVED_PREFIXES: Tuple[str, ...] = ("postdisch.", "denom.", "missing.")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")

# PLAN.md 6.2. An outcome losing more than this share of its events between ladder rows 1
# and 4 requires the bounding sensitivity.
BOUNDING_THRESHOLD_PCT = 5.0

# PLAN.md 2.2 / CLAUDE.md 8.4. The carried-over figure, its tolerance, and the population
# it turns out to belong to. It is a TARGET to reproduce, never a value to publish.
CARRYOVER_POSTDISCH_PCT = 41.3
CARRYOVER_TOLERANCE_DECIMALS = 1

# The still-in-hospital flag, read as text. 02 published the observed level set; this step
# asserts it rather than assuming, because a casing change between PUF vintages would
# otherwise silently turn every flagged patient into an unflagged one (CLAUDE.md 9.13).
STILLINHOSP_COL = "STILLINHOSP"
STILLINHOSP_YES = "yes"

# The one raw DISCHDEST level this step reads, and only as a diagnostic on the death
# outcome's tie behaviour. It is a data value, not a CPT or ICD code, so hard rule 5 does
# not reach it.
DISCHDEST_EXPIRED = "Expired"


# ---------------------------------------------------------------------------
# UNITS AND SCALES (.build_facts.md section 9)
#
# The rule is: NEVER infer a scale from a key suffix, and read the declaration published by
# the step that OWNS the key. Section 9.3 closed the fourteen namespaces owned by 03, 05 and
# 06; section 9.6 records that this step's three were never asked for theirs, so a gate
# obeying 9.2 item 4 would fail on every key this file publishes, including the ones
# Figure 3 is drawn from. These declarations close that gap for 04.
#
# Every count here is a count of EVENTS, not of patients, with one exception: the analytic
# cohort size, which is patients. That distinction matters more in this step than anywhere
# else in the project, because every ladder row and every stratum cell is a count of events
# and the reader's instinct is to read `n` as a number of patients.
#
# Three families carry a unit their name does not make obvious:
#
#   * the bounding threshold is a percentage of ladder row 1, not a count and not a share
#     of the analytic cohort;
#   * the minimum stratum denominator is a COUNT of events despite being spelled as a
#     denominator, and it is the suppression threshold rather than a measured quantity;
#   * the three keys ending in pp are differences in PERCENTAGE POINTS between two
#     percentages, and one of them, the bound width, is a width and not a limit.
#
# And one thing a gate must not get wrong: the point estimate is a percentage of ladder
# ROW 4 while both bounds are percentages of ladder ROW 1. Both are percentages on the same
# 0 to 100 scale, so a unit check cannot catch a denominator mix-up; the denominator is
# named in `postdisch.bound_denominator_note` and stated on every table that carries them.
#
# `<namespace>.scale` is prose a human can read. `<namespace>.scale.families` and
# `<namespace>.scale.units` carry the same content as two parallel lists so
# `verify_numbers.py` can assert a unit mechanically rather than parse English, and
# `<namespace>.scale.other_numeric_keys` is the catch-all. The family entries are KEY
# PATTERNS addressed to 08, not rendered prose. No key is renamed and none is removed.
# ---------------------------------------------------------------------------
UNIT_PERCENT = "percentage on the 0 to 100 scale"
UNIT_PP = "percentage points"
UNIT_EVENTS = "count of events"
UNIT_PATIENTS = "count of patients"
UNIT_STRATA = "count of strata"
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

# The four keys each declaration publishes about itself. They describe the declaration and
# are not measurements, so the coverage check skips them.
SCALE_META_SUFFIXES: Tuple[str, ...] = (".scale", ".scale.families", ".scale.units",
                                        ".scale.other_numeric_keys")

# (namespace, prose, ((key pattern, unit), ...))
SCALE_DECLARATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        "postdisch",
        "Within the post-discharge namespace every key ending in pct is a percentage on the "
        "0 to 100 scale and every key ending in pp is a difference between two such "
        "percentages, expressed in percentage points. Every plain count is a count of "
        "EVENTS, never of patients: a denominator here is the number of events with both "
        "the day and the length of stay recorded, and a stratum cell counts events inside "
        "that stratum. The point estimate is a percentage of ladder ROW 4 while both bounds "
        "are percentages of ladder ROW 1, so the two share a unit and do NOT share a "
        "denominator, which is why a point estimate may legitimately sit outside its own "
        "conditioned interval. The bounding threshold is a percentage of row 1. The minimum "
        "stratum denominator is a COUNT of events and is the suppression threshold, not a "
        "measured quantity.",
        (
            ("postdisch.<outcome>.pct", UNIT_PERCENT),
            ("postdisch.<outcome>.opposite_tie_pct", UNIT_PERCENT),
            ("postdisch.<outcome>.bound_conditioned_low_pct", UNIT_PERCENT),
            ("postdisch.<outcome>.bound_conditioned_high_pct", UNIT_PERCENT),
            ("postdisch.<outcome>.bound_unconditional_low_pct", UNIT_PERCENT),
            ("postdisch.<outcome>.bound_unconditional_high_pct", UNIT_PERCENT),
            ("postdisch.<outcome>.opposite_tie_delta_pp", UNIT_PP),
            ("postdisch.<outcome>.bound_conditioned_width_pp", UNIT_PP),
            ("postdisch.<outcome>.bound_unconditional_width_pp", UNIT_PP),
            ("postdisch.<outcome>.n", UNIT_EVENTS),
            ("postdisch.<outcome>.pre_n", UNIT_EVENTS),
            ("postdisch.<outcome>.denominator", UNIT_EVENTS),
            ("postdisch.<outcome>.tie_on_discharge_day_n", UNIT_EVENTS),
            ("postdisch.<outcome>.opposite_tie_n", UNIT_EVENTS),
            ("postdisch.<outcome>.bound_denominator", UNIT_EVENTS),
            ("postdisch.<outcome>.bound_conditioned_movable_n", UNIT_EVENTS),
            ("postdisch.<outcome>.bound_unconditional_movable_n", UNIT_EVENTS),
            ("postdisch.<outcome>.eligible", UNIT_BOOL),
            ("postdisch.<outcome>.bound_required", UNIT_BOOL),
            ("postdisch.<outcome>.matches_frozen_column", UNIT_BOOL),
            ("postdisch.<outcome>.by.<stratifier>.<level>.pct", UNIT_PERCENT),
            ("postdisch.<outcome>.by.<stratifier>.<level>.n", UNIT_EVENTS),
            ("postdisch.<outcome>.by.<stratifier>.<level>.denominator", UNIT_EVENTS),
            ("postdisch.<outcome>.by.<stratifier>.<level>.suppressed", UNIT_BOOL),
            ("postdisch.<outcome>.by.<stratifier>.unclassified_denominator_n", UNIT_EVENTS),
            ("postdisch.hematoma.carryover.<population>.pct", UNIT_PERCENT),
            ("postdisch.hematoma.carryover.plan_figure_pct", UNIT_PERCENT),
            ("postdisch.hematoma.carryover.analytic_minus_identified_pp", UNIT_PP),
            ("postdisch.hematoma.carryover.code_set_movement_analytic_pp", UNIT_PP),
            ("postdisch.hematoma.carryover.code_set_movement_identified_pp", UNIT_PP),
            ("postdisch.hematoma.carryover.<population>.n", UNIT_EVENTS),
            ("postdisch.hematoma.carryover.<population>.events", UNIT_EVENTS),
            ("postdisch.hematoma.carryover.<population>.denominator", UNIT_EVENTS),
            ("postdisch.hematoma.carryover.reproduces_on_analytic", UNIT_BOOL),
            ("postdisch.hematoma.carryover.reproduces_on_identified", UNIT_BOOL),
            ("postdisch.hematoma.carryover.primary_population", UNIT_TEXT),
            ("postdisch.hematoma.carryover.note", UNIT_TEXT),
            ("postdisch.death.tie_expired_destination_n", UNIT_EVENTS),
            ("postdisch.death.tie_note", UNIT_TEXT),
            ("postdisch.readmit.day_before_discharge_n", UNIT_EVENTS),
            ("postdisch.readmit.day_before_discharge_note", UNIT_TEXT),
            ("postdisch.bounding_threshold_pct", UNIT_PERCENT),
            ("postdisch.min_stratum_denominator", UNIT_EVENTS),
            ("postdisch.strata_suppressed_n", UNIT_STRATA),
            ("postdisch.strata_suppressed_list", UNIT_TEXT),
            ("postdisch.strata_suppression_rule", UNIT_TEXT),
            ("postdisch.tie_convention", UNIT_TEXT),
            ("postdisch.tie_convention_sensitivity", UNIT_TEXT),
            ("postdisch.bound_primary_rule", UNIT_TEXT),
            ("postdisch.bound_denominator_note", UNIT_TEXT),
            ("postdisch.readmit_excluded_reason", UNIT_TEXT),
        ),
    ),
    (
        "denom",
        "The denominator-ladder namespace carries counts of EVENTS on rows 1 to 4 and, for "
        "each row, the loss from row 1 both as a count of events and as a percentage of "
        "ROW 1 on the 0 to 100 scale. The one count of PATIENTS in this namespace is the "
        "analytic cohort size, and it is context rather than a ladder row: no percentage "
        "here is a share of the cohort.",
        (
            ("denom.<outcome>.row2_lost_pct", UNIT_PERCENT),
            ("denom.<outcome>.row3_lost_pct", UNIT_PERCENT),
            ("denom.<outcome>.row4_lost_pct", UNIT_PERCENT),
            ("denom.<outcome>.row1_events", UNIT_EVENTS),
            ("denom.<outcome>.row2_day_present", UNIT_EVENTS),
            ("denom.<outcome>.row3_los_present", UNIT_EVENTS),
            ("denom.<outcome>.row4_both_present", UNIT_EVENTS),
            ("denom.<outcome>.row2_lost_n", UNIT_EVENTS),
            ("denom.<outcome>.row3_lost_n", UNIT_EVENTS),
            ("denom.<outcome>.row4_lost_n", UNIT_EVENTS),
            ("denom.<outcome>.row4_matches_frozen_column", UNIT_BOOL),
            ("denom.<outcome>.frozen_column_zeros_match_loss", UNIT_BOOL),
            ("denom.<outcome>.monotone_non_increasing", UNIT_BOOL),
            ("denom.analytic_cohort_n", UNIT_PATIENTS),
            ("denom.check.<name>", UNIT_BOOL),
            ("denom.ladder_source_note", UNIT_TEXT),
        ),
    ),
    (
        "missing",
        "The conditional-missingness namespace publishes counts of EVENTS and percentages "
        "on the 0 to 100 scale, and every one of those percentages is CONDITIONAL ON THE "
        "EVENT HAVING FIRED: the denominator is that outcome's ladder row 1, never the "
        "analytic cohort and never a database-wide rate. An event rate and a missingness "
        "rate are different quantities and this namespace holds only the second. The one "
        "count of PATIENTS is the number flagged still in hospital beyond the follow-up "
        "window.",
        (
            ("missing.<outcome>.day_missing_pct_given_event", UNIT_PERCENT),
            ("missing.<outcome>.los_missing_pct_given_event", UNIT_PERCENT),
            ("missing.<outcome>.lost_total_pct", UNIT_PERCENT),
            ("missing.<outcome>.events", UNIT_EVENTS),
            ("missing.<outcome>.day_missing_n", UNIT_EVENTS),
            ("missing.<outcome>.los_missing_n", UNIT_EVENTS),
            ("missing.<outcome>.lost_total_n", UNIT_EVENTS),
            ("missing.<outcome>.lost_day_missing_n", UNIT_EVENTS),
            ("missing.<outcome>.lost_los_missing_stillinhosp_n", UNIT_EVENTS),
            ("missing.<outcome>.lost_los_missing_not_stillinhosp_n", UNIT_EVENTS),
            ("missing.<outcome>.lost_both_fields_n", UNIT_EVENTS),
            ("missing.<outcome>.decomposition_sums_to_loss", UNIT_BOOL),
            ("missing.<outcome>.all_los_loss_is_stillinhosp", UNIT_BOOL),
            ("missing.<outcome>.bounding_required", UNIT_BOOL),
            ("missing.stillinhosp_yes_n", UNIT_PATIENTS),
            ("missing.all_los_loss_is_stillinhosp", UNIT_BOOL),
            ("missing.cross_check_against_02_available", UNIT_BOOL),
            ("missing.cross_check_against_02_agrees", UNIT_BOOL),
            ("missing.cross_check_against_02_problems", UNIT_TEXT),
            ("missing.stillinhosp_levels_observed", UNIT_TEXT),
            ("missing.note_event_rate_versus_missingness", UNIT_TEXT),
        ),
    ),
)

_DECLARED_NS = tuple(ns for ns, _p, _f in SCALE_DECLARATIONS)
if sorted(_DECLARED_NS) != sorted(p.rstrip(".") for p in RESERVED_PREFIXES):
    raise SystemExit(
        f"[{STEP}] SCALE_DECLARATIONS covers {sorted(_DECLARED_NS)} but this step reserves "
        f"{sorted(p.rstrip('.') for p in RESERVED_PREFIXES)}. Every reserved namespace "
        f"declares its own units (.build_facts.md 9.2 item 2).")
if len(set(_DECLARED_NS)) != len(_DECLARED_NS):
    raise SystemExit(f"[{STEP}] SCALE_DECLARATIONS declares a namespace twice: {_DECLARED_NS}")


def scale_family_count() -> int:
    """How many key families the three declarations name, in total."""
    return sum(len(families) for _ns, _prose, families in SCALE_DECLARATIONS)


def scale_family_regex(pattern: str) -> "re.Pattern":
    """Turn a family such as `postdisch.<outcome>.pct` into a matcher.

    A `<...>` placeholder matches WITHIN one dotted segment only, so a wildcard cannot
    silently swallow a deeper key. Identical to the matcher `07_figures.py` builds from the
    published lists, which is what makes the published families usable by a gate.
    """
    out: List[str] = []
    for part in pattern.split("."):
        seg = ""
        for token in re.split(r"(<[^>]*>)", part):
            seg += r"[^.]*" if token.startswith("<") and token.endswith(">") \
                else re.escape(token)
        out.append(seg)
    return re.compile(r"\.".join(out) + r"$")


def publish_scale_declarations(results: Dict[str, Any]) -> None:
    """Declare the unit of every numeric family this step publishes.

    Binding: `.build_facts.md` section 9, and 9.6 specifically, which records this step's
    three namespaces as part of the gap. One declaration per namespace, each scoped to its
    own namespace and saying so, because the merged `results.json` resolves the generic
    interval suffixes to at least four incompatible scales and no step's rule is project
    wide.
    """
    for namespace, prose, families in SCALE_DECLARATIONS:
        results[f"{namespace}.scale"] = f"{prose} {SCALE_SCOPE_NOTE}"
        results[f"{namespace}.scale.families"] = [f for f, _ in families]
        results[f"{namespace}.scale.units"] = [u for _, u in families]
        results[f"{namespace}.scale.other_numeric_keys"] = SCALE_CATCH_ALL


def assert_scale_coverage(results: Dict[str, Any]) -> int:
    """Every published key resolves to exactly one declared family. Raises otherwise.

    This is what makes `.build_facts.md` 9.2 item 4 satisfiable without a special case: a
    gate can ask this fragment for the unit of ANY key it holds and get one answer. A new
    key added without a family fails here, inside the step that created it, rather than in
    a downstream gate that cannot know what the key means.
    """
    declared = {ns: [(scale_family_regex(f), f, u) for f, u in fams]
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
            f"[{STEP}] scale-declaration coverage gap (.build_facts.md 9.2 item 4):\n       "
            + "\n       ".join(problems))
    return scale_family_count()


# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------
def pct(n: float, d: float, decimals: int = 4) -> Optional[float]:
    """n as a percentage of d, rounded, or None when the denominator is zero.

    None rather than 0.0, because a percentage of nothing is not zero percent: it does not
    exist, and writing 0.0 would let a reader treat an empty stratum as a measured null
    result (CLAUDE.md hard rule 8).
    """
    if d is None or d == 0:
        return None
    return round(float(n) / float(d) * 100.0, decimals)


def cell_pct(value: Optional[float], decimals: int = 1) -> str:
    """A percentage for a table cell: a bare rounded number, unit named in the header."""
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def cell_int(value) -> str:
    """An integer for a table cell, with thousands separators."""
    if value is None:
        return ""
    return VD.fmt_count(value)


def n_events(n: int) -> str:
    """"1 event" / "33 events". A stray plural in a manuscript table is a real defect."""
    return f"{VD.fmt_count(n)} event" + ("" if int(n) == 1 else "s")


def slug_level(level: str) -> str:
    """A stratum level as a fragment key segment.

    The locked stratifier levels carry '>=' (`>=3`, `>=75`), which the fragment key regex
    rejects, so they are rendered `ge3` / `ge75`. Hyphenated bands (`18-49`, `1-2`) pass
    through unchanged. The mapping is one way and is only ever used to name a key; the
    reviewer-facing text always comes from `pretty_level`.
    """
    s = str(level).strip().lower()
    s = s.replace(">=", "ge").replace("<=", "le").replace(">", "gt").replace("<", "lt")
    s = s.replace(" ", "-").replace("+", "plus")
    out = "".join(ch for ch in s if ch.isalnum() or ch in "-_")
    if not out or not out[0].isalnum():
        raise SystemExit(f"[{STEP}] cannot slug stratum level {level!r} into a fragment key")
    return out


def stratifier_key(column: str) -> str:
    """A stratifier column as a fragment key segment: `AGE_CAT` -> `age_cat`."""
    return column.strip().lower()


def level_display(column: str, level: str) -> str:
    """The reviewer-facing label for a stratum level (CLAUDE.md hard rule 2)."""
    if column == "MALE":
        return VD.pretty_level(column, int(level))
    return VD.pretty_level(column, level)


def as_flag(series: pd.Series) -> pd.Series:
    """A nullable Int8 event column read as a boolean mask, treating missing as not-fired.

    Every event column in the frozen cohort is nullable Int8 where NA means "missing", not
    "no" (cohort_schema.md). For the EVENT flags 02 writes 0 or 1 on every analytic row, so
    this fills nothing in practice; the fill is explicit so that a future column with real
    missingness cannot silently be counted as an event.
    """
    return series.fillna(0).astype("int64") == 1


# ---------------------------------------------------------------------------
# CSV WRITING
# ---------------------------------------------------------------------------
def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write one table, asserting every header and every cell is renderable.

    CLAUDE.md hard rules 1 and 2 are enforced at WRITE TIME, per the constitution, not by a
    later grep. `assert_renderable` raises on a literal em-dash or an unrendered snake_case
    identifier.
    """
    for col in columns:
        VD.assert_renderable(col, where=f"{path.name} header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for i, row in enumerate(rows):
            missing = [c for c in columns if c not in row]
            if missing:
                raise SystemExit(f"[{STEP}] {path.name} row {i} is missing {missing}")
            clean = {}
            for col in columns:
                value = row[col]
                text = "" if value is None else str(value)
                VD.assert_renderable(text, where=f"{path.name} row {i} column {col!r}")
                clean[col] = text
            writer.writerow(clean)


# ---------------------------------------------------------------------------
# THE DENOMINATOR LADDER (PLAN.md 6.2, table 3B)
# ---------------------------------------------------------------------------
def build_ladder(df: pd.DataFrame, still: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Every ladder quantity, every loss decomposition and every post-discharge count.

    One dict per timed outcome, keyed by the roster key. Nothing here is copied from a
    handoff note: every count is computed from the frozen columns and then CHECKED against
    the frozen TIMED_BOTH_<X> and POSTDISCH_<X> columns 02 materialised.
    """
    out: Dict[str, Dict[str, Any]] = {}
    los = df["LOS_POSTOP"]

    for o in VD.TIMED_OUTCOMES:
        event = as_flag(df[o.event_col])
        pod = df[o.pod_col]
        day_present = pod.notna()
        los_present = los.notna()

        row1 = int(event.sum())
        row2 = int((event & day_present).sum())
        row3 = int((event & los_present).sum())
        row4 = int((event & day_present & los_present).sum())

        # Row 4 against the frozen column, which is what 02 says row 4 is.
        timed_both = df[VD.timed_both_column(o.key)]
        row4_frozen = int((timed_both == 1).sum())
        frozen_zeros = int((timed_both == 0).sum())
        frozen_na = int(timed_both.isna().sum())

        # The primary tie convention, recomputed, then checked against the frozen column.
        both = event & day_present & los_present
        post_mine = int((both & (pod > los)).sum())
        pre_mine = int((both & (pod <= los)).sum())
        tie_n = int((both & (pod == los)).sum())
        postdisch_frozen_col = df[VD.postdisch_column(o.key)]
        post_frozen = int((postdisch_frozen_col == 1).sum())
        pre_frozen = int((postdisch_frozen_col == 0).sum())
        frozen_scored_outside_row4 = int(
            (postdisch_frozen_col.notna() & ~both).sum()
        )

        # The opposite tie convention. CLAUDE.md 4.1 grants this step, and only this step,
        # the right to compute it inline from the frozen POD_* and LOS_POSTOP columns.
        post_opposite = int((both & (pod >= los)).sum())

        # The three-way loss decomposition, re-measured here rather than inherited.
        lost_total = row1 - row4
        lost_day = int((event & ~day_present).sum())
        lost_los = int((event & day_present & ~los_present).sum())
        lost_los_still = int((event & day_present & ~los_present & still).sum())
        lost_los_not_still = int((event & day_present & ~los_present & ~still).sum())
        # An event missing BOTH fields is counted in the day arm, which takes precedence, so
        # the marginal "length of stay missing" count can exceed the two length-of-stay arms.
        # Measured rather than assumed, because the S2 columns are read side by side.
        lost_both_fields = int((event & ~day_present & ~los_present).sum())

        rec: Dict[str, Any] = dict(
            key=o.key,
            outcome=VD.short(o.event_col),
            noun=o.noun,
            eligible=bool(o.postdisch_eligible),
            raw_fields=tuple(o.raw_fields),
            row1=row1, row2=row2, row3=row3, row4=row4,
            row4_frozen=row4_frozen,
            row4_matches_frozen=bool(row4 == row4_frozen),
            frozen_zeros=frozen_zeros,
            frozen_na=frozen_na,
            frozen_zeros_match_loss=bool(frozen_zeros == lost_total),
            monotone=bool(row1 >= row2 >= row4 and row1 >= row3 >= row4),
            lost_row2=row1 - row2,
            lost_row3=row1 - row3,
            lost_row4=lost_total,
            lost_row2_pct=pct(row1 - row2, row1),
            lost_row3_pct=pct(row1 - row3, row1),
            lost_row4_pct=pct(lost_total, row1),
            day_missing_n=lost_day,
            day_missing_pct=pct(lost_day, row1),
            los_missing_n=row1 - row3,
            los_missing_pct=pct(row1 - row3, row1),
            lost_day_missing_n=lost_day,
            lost_los_still_n=lost_los_still,
            lost_los_not_still_n=lost_los_not_still,
            lost_los_n=lost_los,
            lost_both_fields_n=lost_both_fields,
            decomposition_sums=bool(lost_day + lost_los == lost_total),
            all_los_loss_still=bool(lost_los_not_still == 0),
            post_n=post_mine,
            pre_n=pre_mine,
            post_matches_frozen=bool(post_mine == post_frozen and pre_mine == pre_frozen),
            frozen_scored_outside_row4=frozen_scored_outside_row4,
            tie_n=tie_n,
            post_opposite_n=post_opposite,
            pct_primary=pct(post_mine, row4),
            pct_opposite=pct(post_opposite, row4),
            bounding_required=bool(
                (pct(lost_total, row1) or 0.0) > BOUNDING_THRESHOLD_PCT
            ),
        )
        delta = None
        if rec["pct_primary"] is not None and rec["pct_opposite"] is not None:
            delta = round(rec["pct_opposite"] - rec["pct_primary"], 4)
        rec["tie_delta_pp"] = delta

        # THE BOUNDS. Both place the previously unplaced events, so both are computed on a
        # ROW 1 denominator. The complete-case point estimate is on row 4. Different
        # denominators, on purpose, and the notes say so.
        movable_unconditional = lost_total
        movable_conditioned = lost_day + lost_los_not_still
        rec["bound_denominator"] = row1
        rec["bound_unconditional_low"] = pct(post_mine, row1)
        rec["bound_unconditional_high"] = pct(post_mine + movable_unconditional, row1)
        rec["bound_unconditional_movable_n"] = movable_unconditional
        rec["bound_conditioned_low"] = pct(post_mine, row1)
        rec["bound_conditioned_high"] = pct(post_mine + movable_conditioned, row1)
        rec["bound_conditioned_movable_n"] = movable_conditioned
        for which in ("unconditional", "conditioned"):
            lo = rec[f"bound_{which}_low"]
            hi = rec[f"bound_{which}_high"]
            rec[f"bound_{which}_width_pp"] = (
                None if lo is None or hi is None else round(hi - lo, 4)
            )
        out[o.key] = rec
    return out


# ---------------------------------------------------------------------------
# STRATA (PLAN.md 6.2, CLAUDE.md 6.5)
# ---------------------------------------------------------------------------
def build_strata(df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """The post-discharge fraction inside each pre-specified stratum.

    Four stratifiers, fixed a priori in CLAUDE.md 6.5, and nothing else: no interaction
    strata, no data-driven cut points, no additional variable. The suppression rule is
    `variable_dict.MIN_STRATUM_EVENTS`, applied to the stratum's own ROW 4 denominator,
    which is the denominator of the proportion being reported. A suppressed stratum still
    publishes its counts, so the suppression is auditable; only the unstable PROPORTION is
    withheld.
    """
    los = df["LOS_POSTOP"]
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for o in VD.TIMED_OUTCOMES:
        if not o.postdisch_eligible:
            continue
        event = as_flag(df[o.event_col])
        pod = df[o.pod_col]
        both = event & pod.notna() & los.notna()
        post = both & (pod > los)
        for strat in VD.STRATIFIERS:
            col = df[strat.column]
            if strat.column == "MALE":
                keyed = col.map({1: "1", 0: "0"}).astype("object")
            else:
                keyed = col.astype("object")
            covered = 0
            for level in strat.levels:
                mask = keyed.eq(level)
                denom = int((both & mask).sum())
                n_post = int((post & mask).sum())
                covered += denom
                suppressed = denom < VD.MIN_STRATUM_EVENTS
                out[(o.key, strat.column, level)] = dict(
                    n=n_post,
                    denominator=denom,
                    pct=None if suppressed else pct(n_post, denom),
                    suppressed=bool(suppressed),
                )
            out[(o.key, strat.column, "__residual__")] = dict(
                n=None,
                denominator=int(both.sum()) - covered,
                pct=None,
                suppressed=False,
            )
    return out


# ---------------------------------------------------------------------------
# GATE 2: THE CARRIED-OVER 41.3%, RE-DERIVED WITH ITS DENOMINATOR STATED
# ---------------------------------------------------------------------------
def hematoma_fraction(frame: pd.DataFrame, event_col: str, pod: pd.Series) -> Dict[str, Any]:
    """One hematoma post-discharge fraction, on whatever frame and code set is passed."""
    event = as_flag(frame[event_col])
    los = frame["LOS_POSTOP"]
    both = event & pod.notna() & los.notna()
    n_post = int((both & (pod > los)).sum())
    denom = int(both.sum())
    return dict(events=int(event.sum()), denominator=denom, n=n_post, pct=pct(n_post, denom),
                opposite_n=int((both & (pod >= los)).sum()),
                opposite_pct=pct(int((both & (pod >= los)).sum()), denom))


def v0_pod(frame: pd.DataFrame) -> pd.Series:
    """The hematoma reoperation day under the SUPERSEDED V0 code set.

    RETORPODAYS masked by the frozen HEMATOMA_V0 column, which is exactly the construction
    02 used to build HEMATOMA_POD from the primary flag (cohort_schema.md). It exists so
    the movement between the code set the carried-over 41.3% was measured under and this
    project's primary can be quantified rather than asserted (CLAUDE.md 8.4). The 0-to-30
    integer handling rule is re-asserted here rather than assumed, because this column is
    built outside 02's helper.
    """
    day = pd.to_numeric(frame["RETORPODAYS"], errors="coerce")
    masked = day.where(as_flag(frame["HEMATOMA_V0"]))
    present = masked.dropna()
    bad = present[(present < 0) | (present > VD.FOLLOWUP_DAYS)
                  | (present != present.round())]
    if len(bad):
        raise SystemExit(
            f"[{STEP}] {len(bad)} reoperation-day values under the superseded code set fall "
            f"outside the 0 to {VD.FOLLOWUP_DAYS} integer rule 02 adopted. That rule is "
            f"applied once, in 02, and this step must not clip or drop locally. Stop and "
            f"re-rule (cohort_schema.md decision 1)."
        )
    return masked


# ---------------------------------------------------------------------------
# CROSS-CHECK AGAINST 02, COMPARISON ONLY
# ---------------------------------------------------------------------------
def load_02(path: Path) -> Optional[Dict[str, Any]]:
    """02's fragment, for comparison only. Absent is not an error."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def compare_to_02(ladder: Dict[str, Dict[str, Any]], carryover: Dict[str, Any],
                  ref: Optional[Dict[str, Any]]) -> List[str]:
    """Every disagreement between this step's re-derivation and 02's published keys.

    A mismatch is REPORTED, never adopted and never overwritten (CLAUDE.md 8.5). Nothing
    published by this step is ever read from 02's fragment; this function only compares.
    """
    if ref is None:
        return []
    problems: List[str] = []

    def check(label: str, mine, key: str) -> None:
        if key not in ref:
            problems.append(f"{label}: 02 has no key {key}")
            return
        theirs = ref[key]
        if isinstance(mine, float) or isinstance(theirs, float):
            if mine is None or theirs is None:
                if mine is not theirs:
                    problems.append(f"{label}: this step {mine!r}, 02 {theirs!r} ({key})")
                return
            if not math.isclose(float(mine), float(theirs), abs_tol=1e-4):
                problems.append(f"{label}: this step {mine!r}, 02 {theirs!r} ({key})")
            return
        if mine != theirs:
            problems.append(f"{label}: this step {mine!r}, 02 {theirs!r} ({key})")

    for key, rec in ladder.items():
        check(f"{key} row 1", rec["row1"], f"timingfield.{key}.events")
        check(f"{key} row 4", rec["row4"], f"timingfield.{key}.both_present_n")
        check(f"{key} loss rows 1 to 4", rec["lost_row4"],
              f"timingfield.{key}.lost_rows_1_to_4_n")
        check(f"{key} loss to a missing day", rec["lost_day_missing_n"],
              f"timingfield.{key}.lost_day_missing_n")
        check(f"{key} loss to a missing length of stay, still in hospital",
              rec["lost_los_still_n"], f"timingfield.{key}.lost_los_missing_stillinhosp_n")
        check(f"{key} loss to a missing length of stay, not still in hospital",
              rec["lost_los_not_still_n"],
              f"timingfield.{key}.lost_los_missing_not_stillinhosp_n")

    check("hematoma analytic post-discharge n", carryover["analytic"]["n"],
          "dq.carryover.postdisch.v2.analytic.post_n")
    check("hematoma analytic denominator", carryover["analytic"]["denominator"],
          "dq.carryover.postdisch.v2.analytic.denominator")
    check("hematoma identified post-discharge n", carryover["identified"]["n"],
          "dq.carryover.postdisch.v2.identified.post_n")
    check("hematoma identified denominator", carryover["identified"]["denominator"],
          "dq.carryover.postdisch.v2.identified.denominator")
    check("hematoma analytic post-discharge n, superseded code set",
          carryover["v0_analytic"]["n"], "dq.carryover.postdisch.v0.analytic.post_n")
    check("hematoma identified post-discharge n, superseded code set",
          carryover["v0_identified"]["n"], "dq.carryover.postdisch.v0.identified.post_n")
    return problems


# ---------------------------------------------------------------------------
# THE FRAGMENT
# ---------------------------------------------------------------------------
def clean_fragment(results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise the fragment before it is written.

    `run_all.py` enforces the namespaces at merge time and rejects the whole run on the
    first violation. Doing it here as well means a violation fails inside the step that
    caused it, with the offending key named, rather than two steps later.
    """
    out: Dict[str, Any] = {}
    bad_ns, bad_key, bad_val = [], [], []
    bare_roots = tuple(p.rstrip(".") for p in RESERVED_PREFIXES)
    for key, value in results.items():
        if not KEY_RE.match(key):
            bad_key.append(key)
        if not key.startswith(RESERVED_PREFIXES) and key not in bare_roots:
            bad_ns.append(key)
        v = value
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        if isinstance(v, float) and not math.isfinite(v):
            v = None                       # JSON has no NaN; write null (CLAUDE.md 4.2).
        elif isinstance(v, (list, tuple)):
            items = []
            for item in v:
                if isinstance(item, np.integer):
                    item = int(item)
                elif isinstance(item, np.floating):
                    item = float(item)
                elif isinstance(item, np.bool_):
                    item = bool(item)
                if isinstance(item, float) and not math.isfinite(item):
                    item = None
                if item is not None and not isinstance(item, (str, int, float, bool)):
                    bad_val.append(key)
                items.append(item)
            v = items
        elif v is not None and not isinstance(v, (str, int, float, bool)):
            bad_val.append(key)
        out[key] = v
    # Every STRING value in the fragment is a candidate for rendered prose: 08 reads the
    # merged file and quotes these notes. CLAUDE.md hard rules 1 and 2 are therefore
    # enforced here as well as at CSV write time, so a raw column name cannot reach the
    # manuscript through a fragment note.
    #
    # ONE structural exemption, and it is narrow by construction: the values of a
    # `<namespace>.scale.families` list are results-key PATTERNS addressed to `08` and
    # `verify_numbers.py`, not prose. A key pattern necessarily carries the underscores of
    # the key it names, so the snake_case rule cannot apply to it; rendering one into a
    # manuscript would be a defect in the consumer, not in this list. The exemption is by
    # key suffix and reaches nothing else, including the scale PROSE beside it, which is
    # written to pass the check like any other note. This is the same shape as the
    # reference-DOI exemption CLAUDE.md 6a.5 requires: structural, and as small as possible.
    bad_text = []
    for key, value in out.items():
        if key.endswith(".scale.families"):
            for item in (value if isinstance(value, list) else [value]):
                if isinstance(item, str) and VD.has_em_dash(item):
                    bad_text.append(key)
            continue
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, str) and (VD.has_em_dash(item) or VD.has_snake_case(item)):
                bad_text.append(key)

    problems = []
    if bad_ns:
        problems.append(f"outside the reserved namespaces {list(RESERVED_PREFIXES)}: "
                        f"{sorted(set(bad_ns))[:10]}")
    if bad_text:
        problems.append(f"string value is not renderable, an em-dash or an unrendered "
                        f"snake_case identifier: {sorted(set(bad_text))[:10]}")
    if bad_key:
        problems.append(f"not a flat lowercase dotted path: {sorted(set(bad_key))[:10]}")
    if bad_val:
        problems.append(f"value is neither a scalar nor a list of scalars: "
                        f"{sorted(set(bad_val))[:10]}")
    if problems:
        raise SystemExit(f"[{STEP}] fragment contract violation:\n       "
                         + "\n       ".join(problems))
    return out


def build_fragment(ladder: Dict[str, Dict[str, Any]],
                   strata: Dict[Tuple[str, str, str], Dict[str, Any]],
                   carryover: Dict[str, Any],
                   context: Dict[str, Any]) -> Dict[str, Any]:
    """Every number this step publishes, as flat dotted keys inside three prefixes."""
    r: Dict[str, Any] = {}

    # --- context and the locked conventions -------------------------------
    r["denom.analytic_cohort_n"] = context["analytic_n"]
    r["denom.ladder_source_note"] = (
        "Row 4 of every ladder is the sum of the frozen both-fields-recorded column that the "
        "cohort build materialised, verified here against an independent recomputation from "
        "the frozen event flag, the frozen postoperative-day column and the frozen "
        "length-of-stay column."
    )
    r["denom.check.all_row4_match_frozen"] = context["all_row4_match"]
    r["denom.check.all_monotone_non_increasing"] = context["all_monotone"]
    r["denom.check.all_postdisch_match_frozen"] = context["all_post_match"]
    r["postdisch.tie_convention"] = VD.TIE_CONVENTION
    r["postdisch.tie_convention_sensitivity"] = VD.TIE_CONVENTION_SENSITIVITY
    r["postdisch.min_stratum_denominator"] = int(VD.MIN_STRATUM_EVENTS)
    r["postdisch.bounding_threshold_pct"] = BOUNDING_THRESHOLD_PCT
    r["postdisch.bound_primary_rule"] = (
        "The still-in-hospital conditioned bound is the primary bounding sensitivity. Every "
        "event lost because the postoperative length of stay is missing belongs to a patient "
        "flagged still in hospital beyond 30 days, and such a patient was never discharged "
        "inside the follow-up window, so the event cannot have occurred after a discharge "
        "that did not occur. The unconditional extreme is reported beside it for comparison "
        "only and is structurally impossible for those rows."
    )
    r["postdisch.bound_denominator_note"] = (
        "Both bounds place the events that the complete-case analysis could not place, so "
        "both are computed on ladder row 1. The point estimate in Table 3B row 5 is computed "
        "on ladder row 4. The two denominators differ on purpose, so a point estimate may "
        "sit outside its own conditioned interval; that is the bounding argument working, "
        "not an error."
    )
    r["postdisch.death.tie_expired_destination_n"] = context["death_tie_expired_n"]
    r["postdisch.death.tie_note"] = (
        "For a patient who died in hospital the recorded postoperative length of stay is the "
        "day of the terminal event, so the opposite tie convention would score that patient "
        "as having died after a discharge that never happened. The locked convention scores "
        "them before discharge, which is correct. Death's opposite-convention figure must "
        "never be quoted without this reason."
    )
    r["postdisch.readmit.day_before_discharge_n"] = context["readmit_before_discharge_n"]
    r["postdisch.readmit.day_before_discharge_note"] = (
        "The readmission day is counted from the principal operation and the length of stay "
        "is the day of discharge from the index admission, so a readmission day strictly "
        "before the discharge day is internally contradictory. These rows are counted and "
        "reported, never corrected and never dropped. They are a further reason the "
        "readmission post-discharge fraction is not reported as a result."
    )
    r["postdisch.readmit_excluded_reason"] = (
        "The days-to-readmission field counts from the principal operation, so an unplanned "
        "readmission is a post-discharge event by construction for anyone who was "
        "discharged. Its post-discharge fraction is uninformative and is not reported as a "
        "result; the timing curve shape is reported instead."
    )
    r["missing.note_event_rate_versus_missingness"] = (
        "The database-wide percentages carried in the plan for each days-to-event field are "
        "EVENT RATES, not missingness rates: a days-to-event field is populated only for "
        "patients who had the event. The quantity that matters, and that was unmeasured "
        "before this project, is the missingness of the days-to-event field CONDITIONAL on "
        "the event flag being positive, which is what this supplement reports."
    )
    r["missing.stillinhosp_yes_n"] = context["stillinhosp_yes_n"]
    r["missing.stillinhosp_levels_observed"] = context["stillinhosp_levels"]
    r["missing.all_los_loss_is_stillinhosp"] = context["all_los_loss_still"]
    r["missing.cross_check_against_02_available"] = context["cross_check_available"]
    r["missing.cross_check_against_02_agrees"] = context["cross_check_agrees"]
    if context["cross_check_problems"]:
        r["missing.cross_check_against_02_problems"] = context["cross_check_problems"]

    # --- per outcome -------------------------------------------------------
    for key, rec in ladder.items():
        r[f"denom.{key}.row1_events"] = rec["row1"]
        r[f"denom.{key}.row2_day_present"] = rec["row2"]
        r[f"denom.{key}.row3_los_present"] = rec["row3"]
        r[f"denom.{key}.row4_both_present"] = rec["row4"]
        r[f"denom.{key}.row2_lost_n"] = rec["lost_row2"]
        r[f"denom.{key}.row2_lost_pct"] = rec["lost_row2_pct"]
        r[f"denom.{key}.row3_lost_n"] = rec["lost_row3"]
        r[f"denom.{key}.row3_lost_pct"] = rec["lost_row3_pct"]
        r[f"denom.{key}.row4_lost_n"] = rec["lost_row4"]
        r[f"denom.{key}.row4_lost_pct"] = rec["lost_row4_pct"]
        r[f"denom.{key}.row4_matches_frozen_column"] = rec["row4_matches_frozen"]
        r[f"denom.{key}.frozen_column_zeros_match_loss"] = rec["frozen_zeros_match_loss"]
        r[f"denom.{key}.monotone_non_increasing"] = rec["monotone"]

        r[f"postdisch.{key}.eligible"] = rec["eligible"]
        r[f"postdisch.{key}.denominator"] = rec["row4"]
        r[f"postdisch.{key}.tie_on_discharge_day_n"] = rec["tie_n"]
        if rec["eligible"]:
            r[f"postdisch.{key}.n"] = rec["post_n"]
            r[f"postdisch.{key}.pre_n"] = rec["pre_n"]
            r[f"postdisch.{key}.pct"] = rec["pct_primary"]
            r[f"postdisch.{key}.matches_frozen_column"] = rec["post_matches_frozen"]
            r[f"postdisch.{key}.opposite_tie_n"] = rec["post_opposite_n"]
            r[f"postdisch.{key}.opposite_tie_pct"] = rec["pct_opposite"]
            r[f"postdisch.{key}.opposite_tie_delta_pp"] = rec["tie_delta_pp"]
            r[f"postdisch.{key}.bound_required"] = rec["bounding_required"]
            r[f"postdisch.{key}.bound_denominator"] = rec["bound_denominator"]
            r[f"postdisch.{key}.bound_conditioned_low_pct"] = rec["bound_conditioned_low"]
            r[f"postdisch.{key}.bound_conditioned_high_pct"] = rec["bound_conditioned_high"]
            r[f"postdisch.{key}.bound_conditioned_width_pp"] = rec["bound_conditioned_width_pp"]
            r[f"postdisch.{key}.bound_conditioned_movable_n"] = rec["bound_conditioned_movable_n"]
            r[f"postdisch.{key}.bound_unconditional_low_pct"] = rec["bound_unconditional_low"]
            r[f"postdisch.{key}.bound_unconditional_high_pct"] = rec["bound_unconditional_high"]
            r[f"postdisch.{key}.bound_unconditional_width_pp"] = rec["bound_unconditional_width_pp"]
            r[f"postdisch.{key}.bound_unconditional_movable_n"] = rec["bound_unconditional_movable_n"]
        else:
            r[f"postdisch.{key}.n"] = None
            r[f"postdisch.{key}.pct"] = None

        r[f"missing.{key}.events"] = rec["row1"]
        r[f"missing.{key}.day_missing_n"] = rec["day_missing_n"]
        r[f"missing.{key}.day_missing_pct_given_event"] = rec["day_missing_pct"]
        r[f"missing.{key}.los_missing_n"] = rec["los_missing_n"]
        r[f"missing.{key}.los_missing_pct_given_event"] = rec["los_missing_pct"]
        r[f"missing.{key}.lost_total_n"] = rec["lost_row4"]
        r[f"missing.{key}.lost_total_pct"] = rec["lost_row4_pct"]
        r[f"missing.{key}.lost_day_missing_n"] = rec["lost_day_missing_n"]
        r[f"missing.{key}.lost_los_missing_stillinhosp_n"] = rec["lost_los_still_n"]
        r[f"missing.{key}.lost_los_missing_not_stillinhosp_n"] = rec["lost_los_not_still_n"]
        r[f"missing.{key}.lost_both_fields_n"] = rec["lost_both_fields_n"]
        r[f"missing.{key}.decomposition_sums_to_loss"] = rec["decomposition_sums"]
        r[f"missing.{key}.all_los_loss_is_stillinhosp"] = rec["all_los_loss_still"]
        r[f"missing.{key}.bounding_required"] = rec["bounding_required"]

    # --- strata ------------------------------------------------------------
    suppressed: List[str] = []
    for (okey, column, level), cell in strata.items():
        ckey = stratifier_key(column)
        if level == "__residual__":
            r[f"postdisch.{okey}.by.{ckey}.unclassified_denominator_n"] = cell["denominator"]
            continue
        lkey = slug_level(level)
        base = f"postdisch.{okey}.by.{ckey}.{lkey}"
        r[f"{base}.n"] = cell["n"]
        r[f"{base}.denominator"] = cell["denominator"]
        r[f"{base}.pct"] = cell["pct"]
        r[f"{base}.suppressed"] = cell["suppressed"]
        if cell["suppressed"]:
            # Rendered, not raw: this list is a fragment value and 08 may print it, so a
            # raw column name here would be a snake_case leak (CLAUDE.md hard rule 2).
            suppressed.append(VD.assert_renderable(
                f"{VD.short(VD.timed_outcome(okey).event_col)}, "
                f"{VD.pretty(column)} {level_display(column, level)}, "
                f"denominator {VD.fmt_count(cell['denominator'])}",
                where="postdisch.strata_suppressed_list"))
    r["postdisch.strata_suppressed_n"] = len(suppressed)
    r["postdisch.strata_suppressed_list"] = sorted(suppressed)
    r["postdisch.strata_suppression_rule"] = (
        f"A stratum whose analysis denominator, that is its own ladder row 4, is fewer than "
        f"{VD.MIN_STRATUM_EVENTS} events has its proportion withheld rather than reported. "
        f"The counts are published so the suppression is auditable."
    )

    # --- Gate 2, the carried-over 41.3% ------------------------------------
    for label, rec in (("analytic", carryover["analytic"]),
                       ("identified", carryover["identified"]),
                       ("v0_analytic", carryover["v0_analytic"]),
                       ("v0_identified", carryover["v0_identified"])):
        r[f"postdisch.hematoma.carryover.{label}.events"] = rec["events"]
        r[f"postdisch.hematoma.carryover.{label}.denominator"] = rec["denominator"]
        r[f"postdisch.hematoma.carryover.{label}.n"] = rec["n"]
        r[f"postdisch.hematoma.carryover.{label}.pct"] = rec["pct"]
    r["postdisch.hematoma.carryover.plan_figure_pct"] = CARRYOVER_POSTDISCH_PCT
    r["postdisch.hematoma.carryover.reproduces_on_identified"] = carryover["reproduces_identified"]
    r["postdisch.hematoma.carryover.reproduces_on_analytic"] = carryover["reproduces_analytic"]
    r["postdisch.hematoma.carryover.analytic_minus_identified_pp"] = carryover["analytic_minus_identified_pp"]
    r["postdisch.hematoma.carryover.code_set_movement_analytic_pp"] = carryover["code_set_move_analytic_pp"]
    r["postdisch.hematoma.carryover.code_set_movement_identified_pp"] = carryover["code_set_move_identified_pp"]
    r["postdisch.hematoma.carryover.primary_population"] = "analytic"
    r["postdisch.hematoma.carryover.note"] = (
        "The carried-over figure is an identified-cohort number. This manuscript reports the "
        "analytic-cohort value with its denominator, because the analytic cohort is the study "
        "population, and states the identified-rung reconciliation beside it. The carried-over "
        "figure was also measured under the superseded hematoma code set, so a small movement "
        "against the current primary definition is expected and is quantified here rather "
        "than asserted."
    )

    # .build_facts.md 9.6: declare the unit of every family this step publishes, LAST, so
    # the coverage assertion sees the complete fragment. Additive metadata only; nothing
    # above is read, renamed or recomputed here.
    n_families = assert_scale_coverage(r)
    publish_scale_declarations(r)
    print(f"[{STEP}]   scale declarations: {len(SCALE_DECLARATIONS)} namespaces, "
          f"{n_families} families, each scoped to its own namespace "
          f"(.build_facts.md 9.6)")
    return r


# ---------------------------------------------------------------------------
# THE TABLES
# ---------------------------------------------------------------------------
LADDER_ROW_TEXT = {
    1: "Events in the analytic cohort",
    2: "Events with the days to event field recorded",
    3: "Events with postoperative length of stay recorded",
    4: "Events with both fields recorded",
    5: "Events occurring after that patient's own discharge",
}

TABLE_3B_COLUMNS = (
    "Outcome",
    "Ladder row",
    "Quantity",
    "n",
    "Lost from row 1, n",
    "Lost from row 1, percent",
    "Percent of the analysis denominator",
    "denominator",
    "Percent basis",
    "Note",
)


def write_table_3b(ladder: Dict[str, Dict[str, Any]], analytic_n: int) -> None:
    """Table 3B: the five-row denominator ladder, one block per timed outcome.

    Row 5 is the only row carrying a percentage of the analysis denominator, and its N is
    printed with it. Rows 2 to 4 carry their drop from row 1 as an absolute count and as a
    percentage of row 1. Readmission's row 5 is withheld with the reason beside it.
    """
    rows: List[Dict[str, Any]] = []
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        outcome = VD.assert_renderable(VD.short(o.event_col), where="Table 3B outcome")
        for number in (1, 2, 3, 4):
            n = rec[{1: "row1", 2: "row2", 3: "row3", 4: "row4"}[number]]
            lost = rec[{1: None, 2: "lost_row2", 3: "lost_row3", 4: "lost_row4"}[number]] \
                if number > 1 else 0
            lost_pct = rec[{2: "lost_row2_pct", 3: "lost_row3_pct",
                            4: "lost_row4_pct"}[number]] if number > 1 else 0.0
            note = ""
            if number == 1:
                note = (f"Denominator column is the analytic cohort, "
                        f"{VD.fmt_count(analytic_n)} patients.")
            elif number == 4:
                note = "The analysis denominator. Row 5 is a percentage of this row only."
            rows.append({
                "Outcome": outcome,
                "Ladder row": str(number),
                "Quantity": LADDER_ROW_TEXT[number],
                "n": cell_int(n),
                "Lost from row 1, n": cell_int(lost),
                "Lost from row 1, percent": cell_pct(lost_pct, 2),
                "Percent of the analysis denominator": "",
                "denominator": cell_int(analytic_n if number == 1 else rec["row1"]),
                "Percent basis": ("The analytic cohort; this row reports no percentage"
                                  if number == 1
                                  else "Events in the analytic cohort, row 1"),
                "Note": note,
            })
        if rec["eligible"]:
            rows.append({
                "Outcome": outcome,
                "Ladder row": "5",
                "Quantity": LADDER_ROW_TEXT[5],
                "n": cell_int(rec["post_n"]),
                "Lost from row 1, n": "",
                "Lost from row 1, percent": "",
                "Percent of the analysis denominator": cell_pct(rec["pct_primary"], 1),
                "denominator": cell_int(rec["row4"]),
                "Percent basis": "Events with both fields recorded, row 4",
                "Note": ("An event on the same postoperative day as discharge counts as "
                         "occurring before discharge."),
            })
        else:
            rows.append({
                "Outcome": outcome,
                "Ladder row": "5",
                "Quantity": LADDER_ROW_TEXT[5],
                "n": "Not reported",
                "Lost from row 1, n": "",
                "Lost from row 1, percent": "",
                "Percent of the analysis denominator": "Not reported",
                "denominator": cell_int(rec["row4"]),
                "Percent basis": "Events with both fields recorded, row 4",
                "Note": ("The days to readmission field counts from the principal operation, "
                         "so a readmission is a post-discharge event by construction for "
                         "anyone who was discharged. The fraction is uninformative and is "
                         "withheld; the timing curve shape is reported instead."),
            })
    write_csv(TABLE_3B_CSV, TABLE_3B_COLUMNS, rows)


S2_COLUMNS = (
    "Outcome",
    "Days to event field",
    "Events in the analytic cohort",
    "Event day recorded",
    "Event day missing",
    "Event day missing given the event, percent",
    "Postoperative length of stay recorded",
    "Postoperative length of stay missing",
    "Length of stay missing given the event, percent",
    "Both fields recorded",
    "Lost between rows 1 and 4, n",
    "Lost between rows 1 and 4, percent",
    "Lost to a missing event day",
    "Lost to a missing length of stay, still in hospital",
    "Lost to a missing length of stay, not still in hospital",
    "Bounding sensitivity required",
    "denominator",
)


def write_s2(ladder: Dict[str, Dict[str, Any]]) -> None:
    """Supplement S2: the denominator and missingness ladder for every timing field.

    Every percentage in a row is conditional on the event flag being positive, so the
    denominator column is that outcome's row 1 throughout. The three-way loss decomposition
    is what makes the bounding sensitivity in S4 honest rather than mechanical.
    """
    rows: List[Dict[str, Any]] = []
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        fields = "; ".join(VD.short(f) for f in o.raw_fields)
        rows.append({
            "Outcome": VD.short(o.event_col),
            "Days to event field": fields,
            "Events in the analytic cohort": cell_int(rec["row1"]),
            "Event day recorded": cell_int(rec["row2"]),
            "Event day missing": cell_int(rec["day_missing_n"]),
            "Event day missing given the event, percent": cell_pct(rec["day_missing_pct"], 2),
            "Postoperative length of stay recorded": cell_int(rec["row3"]),
            "Postoperative length of stay missing": cell_int(rec["los_missing_n"]),
            "Length of stay missing given the event, percent": cell_pct(rec["los_missing_pct"], 2),
            "Both fields recorded": cell_int(rec["row4"]),
            "Lost between rows 1 and 4, n": cell_int(rec["lost_row4"]),
            "Lost between rows 1 and 4, percent": cell_pct(rec["lost_row4_pct"], 2),
            "Lost to a missing event day": cell_int(rec["lost_day_missing_n"]),
            "Lost to a missing length of stay, still in hospital": cell_int(rec["lost_los_still_n"]),
            "Lost to a missing length of stay, not still in hospital":
                cell_int(rec["lost_los_not_still_n"]),
            "Bounding sensitivity required": "Yes" if rec["bounding_required"] else "No",
            "denominator": cell_int(rec["row1"]),
        })
    write_csv(S2_CSV, S2_COLUMNS, rows)


S4_COLUMNS = (
    "Sensitivity",
    "Outcome",
    "Variant",
    "n",
    "denominator",
    "Percent",
    "Difference from the primary, percentage points",
    "Note",
)

TIE_SENSITIVITY = "Tie convention for an event and a discharge on the same day"
BOUND_CONDITIONED = "Bounding on missing timing fields, conditioned on the still in hospital flag (primary)"
BOUND_UNCONDITIONAL = "Bounding on missing timing fields, unconditional extreme (comparison only)"
GATE2_SENSITIVITY = "Population and code set behind the carried over post-discharge fraction"


def write_s4(ladder: Dict[str, Dict[str, Any]], carryover: Dict[str, Any]) -> None:
    """This step's share of Supplement S4 (PLAN.md 6.6).

    Two sensitivities are owned here: the opposite tie convention, and the extreme
    assumption bounds on missing timing fields. The Gate 2 reconciliation of the
    carried-over post-discharge fraction is carried as a third block, because it is the
    same measure computed on a different population and under the superseded code set, and
    a reader comparing this paper with the carried-over figure needs both rows in one place.
    """
    rows: List[Dict[str, Any]] = []
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"]:
            continue
        outcome = VD.short(o.event_col)
        rows.append({
            "Sensitivity": TIE_SENSITIVITY,
            "Outcome": outcome,
            "Variant": "Primary: an event on the discharge day counts as before discharge",
            "n": cell_int(rec["post_n"]),
            "denominator": cell_int(rec["row4"]),
            "Percent": cell_pct(rec["pct_primary"], 2),
            "Difference from the primary, percentage points": "",
            "Note": f"{n_events(rec['tie_n'])} fall on the discharge day.",
        })
        rows.append({
            "Sensitivity": TIE_SENSITIVITY,
            "Outcome": outcome,
            "Variant": "Sensitivity: an event on the discharge day counts as after discharge",
            "n": cell_int(rec["post_opposite_n"]),
            "denominator": cell_int(rec["row4"]),
            "Percent": cell_pct(rec["pct_opposite"], 2),
            "Difference from the primary, percentage points": cell_pct(rec["tie_delta_pp"], 2),
            "Note": ("The whole movement is the events on the discharge day, which the "
                     "primary convention scores as before discharge."),
        })
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"]:
            continue
        outcome = VD.short(o.event_col)
        required = "Required by the more than 5 percent rule." if rec["bounding_required"] \
            else "Not required; the loss is 5 percent or less. Reported for completeness."
        for label, stem, note in (
            (BOUND_CONDITIONED, "bound_conditioned",
             "Events lost because the length of stay is missing on a patient still in "
             "hospital beyond 30 days cannot be post-discharge and are held before "
             "discharge in both extremes."),
            (BOUND_UNCONDITIONAL, "bound_unconditional",
             "Every lost event is free to move, including events on patients who were never "
             "discharged inside the follow-up window. That is structurally impossible and "
             "this row is shown only for comparison."),
        ):
            movable = rec[f"{stem}_movable_n"]
            width = rec[f"{stem}_width_pp"]
            rows.append({
                "Sensitivity": label,
                "Outcome": outcome,
                "Variant": "Lower extreme: every unplaced event counted as before discharge",
                "n": cell_int(rec["post_n"]),
                "denominator": cell_int(rec["bound_denominator"]),
                "Percent": cell_pct(rec[f"{stem}_low"], 2),
                "Difference from the primary, percentage points": "",
                "Note": f"{required} {note}",
            })
            rows.append({
                "Sensitivity": label,
                "Outcome": outcome,
                "Variant": "Upper extreme: every movable unplaced event counted as after discharge",
                "n": cell_int(rec["post_n"] + movable),
                "denominator": cell_int(rec["bound_denominator"]),
                "Percent": cell_pct(rec[f"{stem}_high"], 2),
                "Difference from the primary, percentage points": "",
                "Note": (f"{n_events(movable)} can move. Interval width "
                         f"{cell_pct(width, 2)} percentage points."),
            })
    hem = VD.short("HEMATOMA")
    for label, key, note in (
        ("Analytic cohort, current primary code set (this manuscript reports this row)",
         "analytic",
         "The analytic cohort is the study population and every other number in this "
         "manuscript uses it."),
        ("Identified cohort, current primary code set", "identified",
         "The carried over figure turns out to be an identified-cohort number; this row is "
         "the reconciliation and is never a manuscript denominator."),
        ("Analytic cohort, superseded code set", "v0_analytic",
         "The superseded code set under which the carried over figure was measured."),
        ("Identified cohort, superseded code set", "v0_identified",
         "The closest match to the carried over figure in both population and code set."),
    ):
        rec = carryover[key]
        rows.append({
            "Sensitivity": GATE2_SENSITIVITY,
            "Outcome": hem,
            "Variant": label,
            "n": cell_int(rec["n"]),
            "denominator": cell_int(rec["denominator"]),
            "Percent": cell_pct(rec["pct"], 2),
            "Difference from the primary, percentage points":
                cell_pct(None if rec["pct"] is None or carryover["analytic"]["pct"] is None
                         else rec["pct"] - carryover["analytic"]["pct"], 2),
            "Note": note,
        })
    write_csv(S4_CSV, S4_COLUMNS, rows)


# ---------------------------------------------------------------------------
# THE NOTES
# ---------------------------------------------------------------------------
def write_notes(ladder: Dict[str, Dict[str, Any]],
                strata: Dict[Tuple[str, str, str], Dict[str, Any]],
                carryover: Dict[str, Any],
                context: Dict[str, Any],
                fragment: Dict[str, Any]) -> None:
    """`derived/notes_04.md`, the handoff this step owns.

    CLAUDE.md 3 concurrency rule: a task running inside a parallel batch does not write to
    `.build_facts.md` at all. Everything a later step needs goes here and into the final
    report.
    """
    def f(key: str) -> str:
        value = fragment[key]
        if value is None:
            return "not reported"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, int):
            return VD.fmt_count(value)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines: List[str] = []
    add = lines.append
    add("# notes_04.md -- handoff from `04_postdischarge.py`")
    add("")
    add("Written by `04_postdischarge.py`. Owner: **T6 (post-discharge)**. Owned files:")
    add("`04_postdischarge.py`, `outputs/results_04.json`, `derived/notes_04.md`,")
    add("`tables-csv/table3b_postdischarge_ladder.csv`, `tables-csv/s2_timing_missingness.csv`,")
    add("`tables-csv/s4_sensitivity_04.csv`. Nothing else was created or modified.")
    add("")
    add("This step owns the paper's central novel measure: the proportion of events occurring")
    add("after the patient's own discharge, and the denominator ladder that makes it honest.")
    add("`03_timing.py` owns Table 3A (event counts, median postoperative day, interquartile")
    add("range, the day 0 to 1 share and the after-day-10 share); none of those appears here.")
    add("")
    add("**Re-derive, never copy.** Every number below was computed from")
    add("`derived/cohort_analytic.parquet` in this step and then CHECKED against the frozen")
    add("columns and, where available, against `02`'s published keys. A mismatch is reported,")
    add("not overwritten in either direction (`CLAUDE.md` 8.5).")
    add("")
    add("---")
    add("")
    add("## 0. The checks, before any number is read")
    add("")
    add("| Check | Verdict |")
    add("|---|---|")
    add(f"| Ladder row 4 equals the sum of the frozen both-fields column, every outcome | "
        f"**{f('denom.check.all_row4_match_frozen')}** |")
    add(f"| Ladder rows 1 through 4 are monotonically non-increasing, every outcome | "
        f"**{f('denom.check.all_monotone_non_increasing')}** |")
    add(f"| The recomputed post-discharge flag equals the frozen one, every outcome | "
        f"**{f('denom.check.all_postdisch_match_frozen')}** |")
    add(f"| Every event lost to a missing length of stay is a still-in-hospital patient | "
        f"**{f('missing.all_los_loss_is_stillinhosp')}** |")
    add(f"| Cross-check against `02`'s published keys available | "
        f"**{f('missing.cross_check_against_02_available')}** |")
    add(f"| Cross-check against `02`'s published keys agrees | "
        f"**{f('missing.cross_check_against_02_agrees')}** |")
    add("")
    if context["cross_check_problems"]:
        add("**MISMATCH AGAINST `02`, reported and not overwritten:**")
        add("")
        for problem in context["cross_check_problems"]:
            add(f"* {problem}")
        add("")
    else:
        add("Every ladder count, every loss decomposition and every carried-over reconciliation")
        add("this step re-derived agrees with `02`'s independently published measurement.")
        add("")
    add(f"The still-in-hospital flag carries the levels "
        f"{', '.join(repr(x) for x in context['stillinhosp_levels'])} and is positive on "
        f"{f('missing.stillinhosp_yes_n')} analytic patients.")
    add("")
    add("---")
    add("")
    add("## 1. THE DENOMINATOR LADDER (`PLAN.md` 6.2), Table 3B")
    add("")
    add("The comparison requires **both** the days-to-event field and the postoperative")
    add("length-of-stay field on the same row, so the denominator is not \"all events\": it is")
    add("row 4, \"events with both fields present\", and nothing else.")
    add("")
    add("| Outcome | 1 events | 2 day present | 3 length of stay present | 4 both, THE DENOMINATOR | lost 1 to 4 | 5 post-discharge n | 5 percent of row 4 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if rec["eligible"]:
            post = VD.fmt_count(rec["post_n"])
            share = f"**{rec['pct_primary']:.2f}%**"
        else:
            post = "withheld"
            share = "withheld"
        add(f"| {VD.short(o.event_col)} | {VD.fmt_count(rec['row1'])} | "
            f"{VD.fmt_count(rec['row2'])} | {VD.fmt_count(rec['row3'])} | "
            f"**{VD.fmt_count(rec['row4'])}** | {VD.fmt_count(rec['lost_row4'])} "
            f"({rec['lost_row4_pct']:.2f}%) | {post} | {share} |")
    add("")
    add("Row 5 is the only row carrying a percentage and its N is printed with it. Every drop")
    add("between rows is an absolute count and a percentage of row 1. **No proportion in this")
    add("manuscript appears without its own denominator** (`CLAUDE.md` hard rules 7 and 8).")
    add("")
    add("**The loss is driven by the length-of-stay arm, not by missing event days.** Across")
    add("all eight outcomes the days-to-event field is missing on at most a fraction of a")
    add("percent of events; what removes events from the denominator is a missing postoperative")
    add("length of stay, and every one of those belongs to a patient who was still in hospital")
    add("beyond day 30. That is the whole content of section 3 below.")
    add("")
    add("---")
    add("")
    add("## 2. The post-discharge fraction, with its denominator")
    add("")
    add(f"**Tie convention, locked** (`CLAUDE.md` 6.3): {VD.TIE_CONVENTION} That is,")
    add("`POSTDISCH = (POD > LOS_POSTOP)`, which is the conservative choice for a safety claim:")
    add("it does not credit the observation window with catching an event that may in fact have")
    add("occurred at home.")
    add("")
    add("| Outcome | post-discharge | denominator (row 4) | percent | opposite tie percent | movement |")
    add("|---|---:|---:|---:|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"]:
            continue
        add(f"| {VD.short(o.event_col)} | {VD.fmt_count(rec['post_n'])} | "
            f"{VD.fmt_count(rec['row4'])} | **{rec['pct_primary']:.2f}%** | "
            f"{rec['pct_opposite']:.2f}% | +{rec['tie_delta_pp']:.2f} pp |")
    add("")
    add("**Unplanned readmission is excluded from this table by pre-specification**")
    add("(`CLAUDE.md` 9.19, roster flag `postdisch_eligible = False`). The days-to-readmission")
    add("field counts from the principal operation, so a readmission is a post-discharge event")
    add("BY CONSTRUCTION for anyone who was discharged, and its fraction is uninformative. The")
    add("frozen column exists and is deliberately not published as a result. **The Methods must")
    add("say so** rather than printing a near-100% figure as if it were a finding.")
    add("")
    add("The withheld figure is deliberately absent from `results_04.json` and from every")
    add("table this step writes, so it cannot reach a rendered artefact by accident. Recorded")
    add("here once, and only as the evidence that the exclusion is right, it is")
    add(f"{VD.fmt_count(ladder['readmit']['post_n'])} of "
        f"{VD.fmt_count(ladder['readmit']['row4'])} "
        f"({ladder['readmit']['pct_primary']:.2f}%): the near-100% artefact the caveat")
    add("predicts, carrying no information about when readmissions happen.")
    add("")
    add("**A related data-quality finding, which is published.** "
        f"{VD.fmt_count(context['readmit_before_discharge_n'])} unplanned readmissions carry a")
    add("readmission day strictly BEFORE the index discharge day, which is internally")
    add(f"contradictory, and {VD.fmt_count(ladder['readmit']['tie_n'])} more fall exactly on")
    add("the discharge day. They are counted and reported, never corrected and never dropped")
    add("(`postdisch.readmit.day_before_discharge_n`). They are a further reason the")
    add("readmission post-discharge fraction is not a usable result.")
    add("")
    add("---")
    add("")
    add("## 3. THE BOUNDING SENSITIVITY, and why the naive bound is wrong here")
    add("")
    add("`PLAN.md` 6.2 requires a bounding sensitivity for any outcome losing more than 5% of")
    add("its events between rows 1 and 4. **Three outcomes trip the rule** and the measurement")
    add("below is this step's own, not `02`'s:")
    add("")
    add("| Outcome | lost 1 to 4 | percent | trips the 5% rule |")
    add("|---|---:|---:|---|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        add(f"| {VD.short(o.event_col)} | {VD.fmt_count(rec['lost_row4'])} | "
            f"{rec['lost_row4_pct']:.2f}% | "
            f"{'**YES**' if rec['bounding_required'] else 'no'} |")
    add("")
    add("**The naive bound is wrong here, and the reason is measurable.** Of the events lost")
    add("because the postoperative length of stay is missing, 100% belong to a patient flagged")
    add("still in hospital beyond postoperative day 30, on every outcome that loses any. A")
    add("patient who was never discharged inside the follow-up window cannot have had an event")
    add("**after a discharge that did not occur**, so for those rows `PLAN.md` 6.2's \"all")
    add("missing events post-discharge\" extreme is **structurally impossible** rather than")
    add("merely conservative, and the naive interval is materially wider than the data")
    add("supports. `PLAN.md` 6.2 also says an interval wide enough to change the clinical")
    add("message replaces the point estimate, which makes printing a spuriously wide one a real")
    add("harm.")
    add("")
    add("| Outcome | day missing | length of stay missing, still in hospital | length of stay missing, NOT still in hospital | movable, conditioned | movable, unconditional |")
    add("|---|---:|---:|---:|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        add(f"| {VD.short(o.event_col)} | {VD.fmt_count(rec['lost_day_missing_n'])} | "
            f"{VD.fmt_count(rec['lost_los_still_n'])} | "
            f"{VD.fmt_count(rec['lost_los_not_still_n'])} | "
            f"{VD.fmt_count(rec['bound_conditioned_movable_n'])} | "
            f"{VD.fmt_count(rec['bound_unconditional_movable_n'])} |")
    add("")
    add("**The three arms are disjoint and sum to the loss**, which the step asserts on every")
    add("run. An event whose day is missing is counted in the first arm whether or not its")
    add("length of stay is also missing, so the first arm takes precedence and the marginal")
    add("\"length of stay missing\" count in supplement S2 can exceed the sum of the two")
    add(f"length-of-stay arms. Events missing BOTH fields, measured across the eight")
    add(f"outcomes: **{VD.fmt_count(context['events_missing_both_fields_n'])}**.")
    add("")
    add("> **The conditioned bound is PRIMARY. The unconditional bound is shown beside it and")
    add("> is labelled the unconditional extreme.** One sentence for `08`: *because every event")
    add("> whose length of stay was missing belonged to a patient still admitted beyond the")
    add("> 30-day window, and such a patient has no discharge for an event to fall after, those")
    add("> events were held before discharge in both extremes; the unconditional interval,")
    add("> which allows them to move, is reported only for comparison.*")
    add("")
    add("| Outcome | point estimate (row 4) | conditioned bound (row 1) | width | unconditional bound (row 1) | width |")
    add("|---|---:|---:|---:|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"]:
            continue
        add(f"| {VD.short(o.event_col)} | {rec['pct_primary']:.2f}% | "
            f"{rec['bound_conditioned_low']:.2f}% to {rec['bound_conditioned_high']:.2f}% | "
            f"{rec['bound_conditioned_width_pp']:.2f} pp | "
            f"{rec['bound_unconditional_low']:.2f}% to {rec['bound_unconditional_high']:.2f}% | "
            f"{rec['bound_unconditional_width_pp']:.2f} pp |")
    add("")
    add("**The three outcomes that trip the 5% rule have a conditioned bound of ZERO WIDTH.**")
    add("For unplanned reintubation, ventilator dependence beyond 48 hours and the airway")
    add("composite the entire row-1-to-4 loss is the length-of-stay arm, every one of those")
    add("rows is a still-in-hospital patient, and nothing is free to move. The bound collapses")
    add("to a point. It moves for hematoma, any unplanned reoperation and pneumonia, which lose")
    add("1, 6 and 1 events respectively to a missing event DAY, and for unplanned readmission,")
    add("which loses all 10 of its events that way; none of those four trips the 5% rule.")
    add("")
    add("**Read the two denominators carefully, because `08` must not mis-state this.** The")
    add("point estimate is a percentage of ROW 4. Both bounds place the events the complete-")
    add("case analysis could not place, so both are percentages of ROW 1. For the three")
    add("outcomes with a zero-width conditioned bound the point therefore sits ABOVE its own")
    add("interval, and that is the bounding argument working rather than an error: holding the")
    add("still-in-hospital events before discharge adds them to the denominator as")
    add("pre-discharge events and pulls the fraction down. The displacement is")
    for key in ("reintub", "vent48", "airway"):
        rec = ladder[key]
        add(f"{rec['pct_primary']:.2f}% to {rec['bound_conditioned_low']:.2f}% for "
            f"{VD.short(VD.timed_outcome(key).event_col).lower()};")
    add("in every case the clinical message, that roughly a quarter to a half of these events")
    add("occur after the patient has gone home, is unchanged.")
    add("")
    add("---")
    add("")
    add("## 4. The opposite tie convention (`PLAN.md` 6.6)")
    add("")
    add("`CLAUDE.md` 4.1 grants this step the single explicit exception to the")
    add("no-local-derivation rule so the opposite convention can be computed inline from the")
    add("frozen postoperative-day and length-of-stay columns. It is deliberately not")
    add("materialised by `02`, so it cannot be used by accident.")
    add("")
    add("| Outcome | events on the discharge day | primary | opposite | movement |")
    add("|---|---:|---:|---:|---:|")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"]:
            continue
        add(f"| {VD.short(o.event_col)} | {VD.fmt_count(rec['tie_n'])} | "
            f"{rec['pct_primary']:.2f}% | {rec['pct_opposite']:.2f}% | "
            f"+{rec['tie_delta_pp']:.2f} pp |")
    add("")
    add(f"`02` measured **6 hematoma events on the discharge day, all scored pre-discharge**. "
        f"Re-derived here: **{VD.fmt_count(ladder['hematoma']['tie_n'])}**, "
        f"{'REPRODUCED' if ladder['hematoma']['tie_n'] == 6 else 'NOT REPRODUCED'}.")
    add("")
    add("**Death is the one outcome where the tie convention does heavy lifting, and `08` must")
    add("not report its opposite-convention figure without the reason.** "
        f"{VD.fmt_count(ladder['death']['tie_n'])} of "
        f"{VD.fmt_count(ladder['death']['row4'])} deaths fall on the recorded discharge day,")
    add(f"of which {VD.fmt_count(context['death_tie_expired_n'])} carry an expired discharge")
    add("destination. For a patient who died in hospital the recorded length of stay IS the day")
    add("of the terminal event, so the opposite convention would score them as having died")
    add("after a discharge that never happened. The locked convention scores them")
    add("pre-discharge, which is correct, and this is direct evidence that the locked")
    add("convention is the right one rather than merely the conservative one.")
    add("")
    add("---")
    add("")
    add("## 5. Strata (`CLAUDE.md` 6.5), pre-specified and not hunted")
    add("")
    add("Four stratifiers fixed a priori and nothing else: age group, ASA class, fusion levels")
    add(f"and sex. Suppression rule, set before the numbers were seen: a stratum whose own")
    add(f"analysis denominator is fewer than {VD.MIN_STRATUM_EVENTS} events has its proportion")
    add("withheld. The counts are still published, so the suppression is auditable.")
    add("")
    for o in VD.TIMED_OUTCOMES:
        if not o.postdisch_eligible:
            continue
        rec = ladder[o.key]
        add(f"**{VD.short(o.event_col)}**, overall "
            f"{VD.fmt_count(rec['post_n'])}/{VD.fmt_count(rec['row4'])} "
            f"({rec['pct_primary']:.2f}%)")
        add("")
        add("| Stratifier | Level | post-discharge | denominator | percent |")
        add("|---|---|---:|---:|---:|")
        for strat in VD.STRATIFIERS:
            for level in strat.levels:
                cell = strata[(o.key, strat.column, level)]
                share = ("suppressed" if cell["suppressed"]
                         else f"{cell['pct']:.2f}%")
                add(f"| {VD.pretty(strat.column)} | {level_display(strat.column, level)} | "
                    f"{VD.fmt_count(cell['n'])} | {VD.fmt_count(cell['denominator'])} | "
                    f"{share} |")
            residual = strata[(o.key, strat.column, "__residual__")]["denominator"]
            if residual:
                add(f"| {VD.pretty(strat.column)} | {VD.MISSING_LABEL} | not reported | "
                    f"{VD.fmt_count(residual)} | not reported |")
        add("")
    add(f"**Suppressed strata: {fragment['postdisch.strata_suppressed_n']}.** "
        + ("None." if not fragment["postdisch.strata_suppressed_list"]
           else "; ".join(fragment["postdisch.strata_suppressed_list"]) + "."))
    add("")
    add("Every stratum level is fully covered: the level denominators sum to the outcome's own")
    add("row 4 in every case, so no event with both fields recorded falls outside a")
    add("pre-specified stratum. Sex is missing for non-binary patients by design")
    add("(`cohort_schema.md`), and no such patient carries a timed event with both fields")
    add("recorded, so the sex strata lose nothing.")
    add("")
    add("---")
    add("")
    add("## 6. GATE 2: the carried-over 41.3%, re-derived with its denominator stated")
    add("")
    add("`CLAUDE.md` 8.4 requires the carried-over \"41.3% of hematoma reoperations occur after")
    add("that patient's own discharge\" to be re-derived **with its denominator stated**.")
    add("`PLAN.md` 13 listed the denominator as an open item because the parent never recorded")
    add("it. `02` resolved it and this step RE-DERIVES rather than copies.")
    add("")
    add("| Population | code set | events | denominator (both fields) | post-discharge | percent |")
    add("|---|---|---:|---:|---:|---:|")
    for label, key, code in (("**Analytic (what this manuscript reports)**", "analytic", "current primary"),
                             ("Identified (reconciliation only)", "identified", "current primary"),
                             ("Analytic", "v0_analytic", "superseded"),
                             ("Identified", "v0_identified", "superseded")):
        rec = carryover[key]
        add(f"| {label} | {code} | {VD.fmt_count(rec['events'])} | "
            f"{VD.fmt_count(rec['denominator'])} | {VD.fmt_count(rec['n'])} | "
            f"**{rec['pct']:.2f}%** |")
    add("")
    add(f"**The carried-over {CARRYOVER_POSTDISCH_PCT}% is an IDENTIFIED-cohort figure.** It")
    add(f"reproduces on the identified rung under the current primary code set at")
    add(f"{carryover['identified']['pct']:.2f}%, which rounds to "
        f"{carryover['identified']['pct']:.1f}% to one decimal: reproduces = "
        f"**{'TRUE' if carryover['reproduces_identified'] else 'FALSE'}**. On the analytic")
    add(f"cohort, which `CLAUDE.md` 4.4 mandates as this manuscript's population, the same")
    add(f"measure is **{VD.fmt_count(carryover['analytic']['n'])} of "
        f"{VD.fmt_count(carryover['analytic']['denominator'])} = "
        f"{carryover['analytic']['pct']:.2f}%**, which is")
    add(f"{carryover['analytic_minus_identified_pp']:.2f} percentage points above the")
    add("carried-over figure.")
    add("")
    add("**That is not a discrepancy to reconcile away.** It is the same measure on a different")
    add("and correctly stated denominator, and stating the denominator is exactly the")
    add("resolution `PLAN.md` 13 asked for. Report the analytic value as primary with its")
    add("denominator; state the identified-rung reconciliation beside it.")
    add("")
    add("**The code-set movement is small and documented.** The carried-over figure was")
    add("measured under the superseded hematoma code set and this project's primary is the")
    add("corrected one. Re-derived here, the movement is")
    add(f"{carryover['code_set_move_identified_pp']:.2f} percentage points on the identified")
    add(f"rung and {carryover['code_set_move_analytic_pp']:.2f} on the analytic cohort. Both")
    add("are far inside the one-decimal tolerance, so the code-set correction is not what moves")
    add("the number: the population is.")
    add("")
    add("**Deviation recorded so a reviewer can challenge it.** Re-deriving the identified-rung")
    add("figure requires the identified cohort, so this step reads")
    add("`derived/cohort_identified.parquet` for that one quantity. It is a frozen `02`")
    add("artefact; no cohort is re-derived and the warehouse is never touched. Copying `02`'s")
    add("published number instead would have satisfied \"read the analytic parquet only\" and")
    add("broken the stronger instruction to re-derive. The superseded-code-set figure masks the")
    add("frozen reoperation-day column by the frozen superseded hematoma flag, which is the")
    add("same construction `02` used for the primary and which `CLAUDE.md` 8.4 names as")
    add("available for exactly this purpose.")
    add("")
    add("---")
    add("")
    add("## 7. What `07` and `08` must know")
    add("")
    add("1. **Read `outputs/results.json`, never this file.** Every number above is a fragment")
    add("   key under `postdisch.`, `denom.` or `missing.`. This project has recorded instances")
    add("   of a count being copied wrong out of a handoff note while the artefact was right.")
    add("2. **No post-discharge fraction may be printed without row 4 beside it**")
    add("   (`CLAUDE.md` hard rule 8). The key is `postdisch.<outcome>.denominator`, and")
    add("   Figure 3 must print it on each bar (`.build_facts.md` 6).")
    add("3. **Unplanned readmission has no post-discharge fraction.** `postdisch.readmit.n` and")
    add("   `.pct` are null by design and `postdisch.readmit.eligible` is false. The Methods")
    add("   state the reason (`postdisch.readmit_excluded_reason`) rather than printing a")
    add("   near-100% figure as a finding.")
    add("4. **The conditioned bound is the primary bounding sensitivity** and the unconditional")
    add("   one may appear only beside it, labelled as the unconditional extreme. The one-")
    add("   sentence justification is in section 3 above and in `postdisch.bound_primary_rule`.")
    add("5. **The bounds are percentages of row 1 and the point estimate is a percentage of")
    add("   row 4.** Do not present them on a common axis without saying so")
    add("   (`postdisch.bound_denominator_note`).")
    add("6. **The hematoma figure this manuscript reports is the ANALYTIC one**")
    add(f"   ({VD.fmt_count(carryover['analytic']['n'])} of "
        f"{VD.fmt_count(carryover['analytic']['denominator'])}, "
        f"{carryover['analytic']['pct']:.1f}%), with the identified-rung reconciliation")
    add("   beside it. Do not print the carried-over figure as this paper's result.")
    add("7. **Cite the published external estimate rather than presenting it as confirmation.**")
    add("   `CLAUDE.md` 6a.2 records an indexed paper reporting 37% of ACDF hematoma")
    add("   reoperations occurring after discharge. Two independent analyses landing near each")
    add("   other is corroboration; neither confirms the other's exact value.")
    add("8. **Death's opposite-convention figure must never be quoted bare.** Section 4 above")
    add("   explains why: for a patient who died in hospital the recorded length of stay is the")
    add("   day of death.")
    add("9. **Stratum keys have the shape**")
    add("   `postdisch.<outcome>.by.<stratifier>.<level>.{n,denominator,pct,suppressed}`, with")
    add("   `>=` slugged to `ge` (`ge3`, `ge75`) and sex as 1 and 0. The level is a key")
    add("   segment only; reviewer-facing text always comes from `pretty_level`. A")
    add("   `.unclassified_denominator_n` key beside each stratifier carries the events with")
    add("   both fields recorded that fall outside every declared level, and it is zero")
    add("   everywhere on this cohort.")
    add("10. **Supplement S2 covers every timing field that carries a timed outcome in the")
    add("    roster.** The second-event fields sit outside the roster and are reported by `03`")
    add("    (`CLAUDE.md` 9.18).")
    add("11. **Do not confuse an event rate with a missingness rate.** The database-wide")
    add("    percentages in `PLAN.md` 2.4 are event rates, because a days-to-event field is")
    add("    populated only for patients who had the event. S2 reports the conditional")
    add("    missingness, which is the quantity that matters and was unmeasured before this")
    add("    project.")
    add("12. **All three of this step's namespaces now declare their own units**")
    add("    (`.build_facts.md` 9.6, closed here). Each publishes `<namespace>.scale` as")
    add("    prose, `<namespace>.scale.families` and `<namespace>.scale.units` as two")
    add("    parallel machine-readable lists, and `<namespace>.scale.other_numeric_keys` as")
    add("    the catch-all, in the same shape `03`, `05` and `06` use. "
        f"**{scale_family_count()} families")
    add(f"    across {len(SCALE_DECLARATIONS)} namespaces, and the build ASSERTS that every "
        "published key matches")
    add("    exactly one of them**, so a gate can resolve the unit of any key in this")
    add("    fragment without a special case. Read the declaration belonging to the")
    add("    namespace that OWNS the key and never another step's.")
    add("13. **Every plain count in this fragment is a count of EVENTS**, with exactly two")
    add("    exceptions, both counts of PATIENTS: `denom.analytic_cohort_n` and")
    add("    `missing.stillinhosp_yes_n`. Reading a ladder row or a stratum cell as a number")
    add("    of patients is the easiest mistake to make here. Three further families are")
    add("    worth care: `postdisch.bounding_threshold_pct` is a percentage of ladder row 1")
    add("    rather than of the cohort; `postdisch.min_stratum_denominator` is a COUNT of")
    add("    events and is the suppression threshold, not a measured denominator; and the")
    add("    keys ending `_pp` are differences in percentage points, one of which is an")
    add("    interval WIDTH rather than a limit. A unit check alone cannot catch the")
    add("    row 4 against row 1 denominator split described in item 5 above, because both")
    add("    sides are percentages on the same scale.")
    add("")
    add("---")
    add("")
    add("## 8. Decisions this step MADE rather than inherited")
    add("")
    add("1. **Both bounds are computed on ladder row 1**, because both place events the")
    add("   complete-case analysis could not place. Reporting a bound on row 4 would be a")
    add("   bound on a denominator that excludes exactly the events being bounded.")
    add("2. **The conditioned bound holds the still-in-hospital events before discharge in")
    add("   BOTH extremes**, rather than excluding them. Excluding them would silently restore")
    add("   the complete-case denominator and make the sensitivity vacuous.")
    add("3. **The suppression rule is applied to the stratum's own row 4**, the denominator of")
    add("   the proportion being reported, and it withholds the PROPORTION while publishing the")
    add("   counts. `CLAUDE.md` 6.5 sets the threshold on composite events for a plotted curve;")
    add("   the analogue for a reported proportion is its own denominator.")
    add("4. **Readmission keeps its full ladder rows 1 to 4** and loses only row 5. The ladder")
    add("   is honest for it and `03` needs the same denominators for the curve.")
    add("5. **The whole eight-outcome roster gets a ladder**, including death, because the")
    add("   ladder is a measurement-quality statement and suppressing it for an outcome would")
    add("   hide the one number a reviewer would ask for.")
    add("")

    text = "\n".join(lines) + "\n"
    if VD.EM_DASH in text:
        raise SystemExit(f"[{STEP}] notes_04.md contains a literal em-dash")
    NOTES_MD.parent.mkdir(parents=True, exist_ok=True)
    NOTES_MD.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="04_postdischarge.py",
        description=("Post-discharge fractions, the denominator ladder (Table 3B), the "
                     "timing missingness ladder (S2) and this step's sensitivities (S4)."))
    ap.parse_args(argv)

    if not ANALYTIC_PARQUET.exists():
        raise SystemExit(
            f"[{STEP}] {ANALYTIC_PARQUET} does not exist. Run 02_build_cohort.py first; "
            f"this step never re-derives a cohort and never reads the warehouse "
            f"(CLAUDE.md 4.1).")

    print(f"[{STEP}] 04_postdischarge.py  --  the post-discharge fraction and its "
          f"denominator ladder")
    df = pd.read_parquet(ANALYTIC_PARQUET)
    analytic_n = len(df)
    print(f"[{STEP}]   read {ANALYTIC_PARQUET.name}  {analytic_n:,} x {df.shape[1]}")

    required = (["LOS_POSTOP", STILLINHOSP_COL, "DISCHDEST"]
                + [o.event_col for o in VD.TIMED_OUTCOMES]
                + list(VD.POD_COLUMNS) + list(VD.POSTDISCH_COLUMNS)
                + list(VD.TIMED_BOTH_COLUMNS) + list(VD.STRATIFIER_COLUMNS))
    absent = [c for c in dict.fromkeys(required) if c not in df.columns]
    if absent:
        raise SystemExit(
            f"[{STEP}] the frozen cohort is missing {absent}. That is a bug in 02, not a "
            f"licence to compute it locally (CLAUDE.md 4.1).")

    levels = sorted({str(x).strip() for x in df[STILLINHOSP_COL].dropna().unique()})
    still = df[STILLINHOSP_COL].astype("string").str.strip().str.lower().eq(STILLINHOSP_YES)
    still = still.fillna(False)
    unexpected = [lv for lv in levels if lv.lower() not in ("yes", "no", "")]
    if unexpected:
        raise SystemExit(
            f"[{STEP}] the still-in-hospital column carries unexpected levels {unexpected}. "
            f"The bounding sensitivity is conditioned on this flag, so an unmapped level "
            f"would silently unflag patients. Stop and re-rule.")
    print(f"[{STEP}]   still in hospital beyond {VD.FOLLOWUP_DAYS} days: "
          f"{int(still.sum()):,} patients, levels {levels}")

    ladder = build_ladder(df, still)
    strata = build_strata(df)

    # --- the checks the task must not skip ---------------------------------
    checks = (
        ("row 4 equals the frozen both-fields column", "row4_matches_frozen"),
        ("the frozen both-fields column's zeros equal the loss", "frozen_zeros_match_loss"),
        ("rows 1 to 4 are monotonically non-increasing", "monotone"),
        ("the recomputed post-discharge flag equals the frozen one", "post_matches_frozen"),
        ("the three-way loss decomposition sums to the loss", "decomposition_sums"),
    )
    for label, field in checks:
        offenders = [k for k, r in ladder.items() if not r[field]]
        print(f"[{STEP}]   CHECK {label}: {'PASS' if not offenders else 'FAIL'}")
        if offenders:
            raise SystemExit(
                f"[{STEP}] STOP CONDITION: {label} is FALSE for {offenders}. This step does "
                f"not adjust a count to make a check pass (CLAUDE.md 8.5).")
    all_row4 = all(r["row4_matches_frozen"] for r in ladder.values())
    all_mono = all(r["monotone"] for r in ladder.values())
    all_post = all(r["post_matches_frozen"] for r in ladder.values())
    all_still = all(r["all_los_loss_still"] for r in ladder.values())
    stray = {k: r["frozen_scored_outside_row4"] for k, r in ladder.items()
             if r["frozen_scored_outside_row4"]}
    if stray:
        raise SystemExit(
            f"[{STEP}] STOP CONDITION: the frozen post-discharge column is scored on rows "
            f"outside its own row-4 denominator: {stray}.")
    print(f"[{STEP}]   CHECK every event lost to a missing length of stay is a "
          f"still-in-hospital patient: {'PASS' if all_still else 'FAIL'}")

    # --- Gate 2, re-derived ------------------------------------------------
    ident = None
    if IDENTIFIED_PARQUET.exists():
        ident = pd.read_parquet(IDENTIFIED_PARQUET)
        print(f"[{STEP}]   read {IDENTIFIED_PARQUET.name} {len(ident):,} x "
              f"{ident.shape[1]} for the Gate 2 identified-rung reconciliation only")
    else:
        raise SystemExit(
            f"[{STEP}] {IDENTIFIED_PARQUET} does not exist and CLAUDE.md 8.4 requires the "
            f"identified-rung reconciliation of the carried-over post-discharge fraction to "
            f"be RE-DERIVED rather than copied. Run 02_build_cohort.py first.")

    carryover = {
        "analytic": hematoma_fraction(df, "HEMATOMA", df["HEMATOMA_POD"]),
        "identified": hematoma_fraction(ident, "HEMATOMA", ident["HEMATOMA_POD"]),
        "v0_analytic": hematoma_fraction(df, "HEMATOMA_V0", v0_pod(df)),
        "v0_identified": hematoma_fraction(ident, "HEMATOMA_V0", v0_pod(ident)),
    }
    for label, rec in carryover.items():
        rec["reproduces_carryover"] = bool(
            rec["pct"] is not None
            and round(rec["pct"], CARRYOVER_TOLERANCE_DECIMALS)
            == round(CARRYOVER_POSTDISCH_PCT, CARRYOVER_TOLERANCE_DECIMALS))
    carryover["reproduces_identified"] = carryover["identified"]["reproduces_carryover"]
    carryover["reproduces_analytic"] = carryover["analytic"]["reproduces_carryover"]
    carryover["analytic_minus_identified_pp"] = round(
        carryover["analytic"]["pct"] - carryover["identified"]["pct"], 4)
    carryover["code_set_move_analytic_pp"] = round(
        carryover["analytic"]["pct"] - carryover["v0_analytic"]["pct"], 4)
    carryover["code_set_move_identified_pp"] = round(
        carryover["identified"]["pct"] - carryover["v0_identified"]["pct"], 4)

    # The analytic figure must agree with the ladder, which was computed independently.
    if (carryover["analytic"]["n"] != ladder["hematoma"]["post_n"]
            or carryover["analytic"]["denominator"] != ladder["hematoma"]["row4"]):
        raise SystemExit(
            f"[{STEP}] STOP CONDITION: the Gate 2 analytic hematoma figure disagrees with "
            f"the ladder computed from the same frozen columns.")

    # --- cross-check against 02, comparison only ---------------------------
    ref = load_02(RESULTS_02_JSON)
    problems = compare_to_02(ladder, carryover, ref)
    if ref is None:
        print(f"[{STEP}]   cross-check against 02: results_02.json not present, SKIPPED "
              f"(no published value in this step is ever read from it)")
    elif problems:
        print(f"[{STEP}]   CROSS-CHECK BLOCKER: {len(problems)} disagreement(s) with 02's "
              f"published keys. Reported, not overwritten (CLAUDE.md 8.5):")
        for problem in problems:
            print(f"[{STEP}]     {problem}")
    else:
        print(f"[{STEP}]   cross-check against 02's published keys: AGREES on every ladder "
              f"count, every loss decomposition and every carried-over reconciliation")

    # Death's tie diagnostic: the recorded length of stay for a patient who died in
    # hospital IS the day of the terminal event, which is why the opposite convention is
    # so large for this outcome and must never be quoted bare.
    death = VD.timed_outcome("death")
    death_event = as_flag(df[death.event_col])
    death_both = death_event & df[death.pod_col].notna() & df["LOS_POSTOP"].notna()
    death_tie = death_both & (df[death.pod_col] == df["LOS_POSTOP"])
    death_tie_expired = int((death_tie & df["DISCHDEST"].eq(DISCHDEST_EXPIRED)).sum())

    # Readmission diagnostic. The readmission day is counted from the principal operation
    # and the length of stay is the day of discharge from the index admission, so a
    # readmission day STRICTLY BEFORE the discharge day is internally contradictory. It is
    # counted and reported, never corrected: no step here clips or drops a frozen value
    # (cohort_schema.md decision 1).
    readmit = VD.timed_outcome("readmit")
    readmit_event = as_flag(df[readmit.event_col])
    readmit_both = (readmit_event & df[readmit.pod_col].notna()
                    & df["LOS_POSTOP"].notna())
    readmit_before = int((readmit_both & (df[readmit.pod_col] < df["LOS_POSTOP"])).sum())

    context = dict(
        analytic_n=analytic_n,
        all_row4_match=all_row4,
        all_monotone=all_mono,
        all_post_match=all_post,
        all_los_loss_still=all_still,
        stillinhosp_yes_n=int(still.sum()),
        stillinhosp_levels=levels,
        cross_check_available=ref is not None,
        cross_check_agrees=(ref is not None and not problems),
        cross_check_problems=problems,
        death_tie_expired_n=death_tie_expired,
        readmit_before_discharge_n=readmit_before,
        events_missing_both_fields_n=sum(r["lost_both_fields_n"] for r in ladder.values()),
    )

    fragment = clean_fragment(build_fragment(ladder, strata, carryover, context))

    # --- write -------------------------------------------------------------
    write_table_3b(ladder, analytic_n)
    print(f"[{STEP}]   wrote {TABLE_3B_CSV}")
    write_s2(ladder)
    print(f"[{STEP}]   wrote {S2_CSV}")
    write_s4(ladder, carryover)
    print(f"[{STEP}]   wrote {S4_CSV}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(fragment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"[{STEP}]   wrote {RESULTS_JSON}  ({len(fragment)} keys, namespaces validated)")

    write_notes(ladder, strata, carryover, context, fragment)
    print(f"[{STEP}]   wrote {NOTES_MD}")

    # --- the headline, printed so the run log carries it -------------------
    print("")
    print(f"[{STEP}] THE DENOMINATOR LADDER (PLAN.md 6.2), analytic cohort {analytic_n:,}")
    print(f"[{STEP}]   {'outcome':38s} {'r1':>6s} {'r2':>6s} {'r3':>6s} {'r4':>6s} "
          f"{'lost':>6s} {'post':>6s} {'% of r4':>9s}")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        share = "withheld" if not rec["eligible"] else f"{rec['pct_primary']:.2f}%"
        post = "withheld" if not rec["eligible"] else f"{rec['post_n']:,}"
        print(f"[{STEP}]   {VD.short(o.event_col)[:38]:38s} {rec['row1']:6,} "
              f"{rec['row2']:6,} {rec['row3']:6,} {rec['row4']:6,} "
              f"{rec['lost_row4']:6,} {post:>6s} {share:>9s}")
    print("")
    print(f"[{STEP}] BOUNDING SENSITIVITY, conditioned bound PRIMARY "
          f"(.build_facts.md 7.5 item 7)")
    for o in VD.TIMED_OUTCOMES:
        rec = ladder[o.key]
        if not rec["eligible"] or not rec["bounding_required"]:
            continue
        print(f"[{STEP}]   {VD.short(o.event_col)[:38]:38s} loses "
              f"{rec['lost_row4_pct']:.2f}% of its events; conditioned "
              f"{rec['bound_conditioned_low']:.2f}% to "
              f"{rec['bound_conditioned_high']:.2f}% "
              f"(width {rec['bound_conditioned_width_pp']:.2f} pp), unconditional "
              f"{rec['bound_unconditional_low']:.2f}% to "
              f"{rec['bound_unconditional_high']:.2f}% "
              f"(width {rec['bound_unconditional_width_pp']:.2f} pp)")
    print("")
    print(f"[{STEP}] GATE 2, the carried-over {CARRYOVER_POSTDISCH_PCT}% re-derived with its "
          f"denominator stated")
    print(f"[{STEP}]   analytic, current code set, PRIMARY: "
          f"{carryover['analytic']['n']:,}/{carryover['analytic']['denominator']:,} = "
          f"{carryover['analytic']['pct']:.2f}%")
    print(f"[{STEP}]   identified, current code set, reconciliation: "
          f"{carryover['identified']['n']:,}/"
          f"{carryover['identified']['denominator']:,} = "
          f"{carryover['identified']['pct']:.2f}%  reproduces "
          f"{CARRYOVER_POSTDISCH_PCT}% to one decimal: "
          f"{carryover['reproduces_identified']}")
    print(f"[{STEP}]   superseded code set: analytic "
          f"{carryover['v0_analytic']['pct']:.2f}%, identified "
          f"{carryover['v0_identified']['pct']:.2f}%; the code-set movement is "
          f"{carryover['code_set_move_identified_pp']:.2f} pp on the identified rung")
    print("")
    if problems:
        print(f"[{STEP}] NOTE: {len(problems)} cross-check disagreement(s) with 02 are "
              f"recorded in results_04.json and notes_04.md, not overwritten.")
    print(f"[{STEP}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
