"""Single source of truth for the ACDF CPT / ICD taxonomy (ACS-NSQIP Adult PUF 2012-2024).

Import from here. Never re-derive CPT or ICD logic anywhere else in the pipeline.
Per `CLAUDE.md` section 5 hard rule, this module is the ONLY place in the project where a
CPT or ICD code may appear as a literal. A bare code literal in any other file is a defect.

The module is pure: no pandas import, no file I/O at import, no printing at import. Every
predicate is total (None / NaN / empty / float-like input returns False rather than raising).
`python3 cpt_acdf.py` runs the self-test and exits 0 on success, non-zero on failure.

===============================================================================
1. PROVENANCE
===============================================================================
Code lists come from `plans/2026-07-30-acdf-hematoma-nsqip.md`, section "Study design".
Every set below was then re-measured read-only against
`nsqip-db/parquet/main/main_2012..2024.parquet` on 2026-07-30, and re-verified on
2026-07-30 during the Batch 1 review. TWO cohort ladders reproduce to the row, and they
differ ONLY in how `AGE` is cast (`CLAUDE.md` section 7):

    | Step                                        | Gate A n | Gate B n | hematoma |
    |---------------------------------------------|----------|----------|----------|
    | principal CPT in PRINCIPAL_CPTS, age >= 18  |  124,266 |  124,344 |      642 |
    | + elective                                  |  121,142 |  121,210 |      613 |
    | - EXCLUSION_CPTS anywhere in OTHERCPT1..10  |  113,411 |  113,469 |      548 |
    | - ASA 5 / ASA missing / preoperative sepsis |  112,585 |  112,642 |      541 |

The hematoma column was RE-PINNED 2026-07-31 to the V2 primary outcome definition, after
three errors were found in the original ICD-10 code set. Every value was re-measured
against the export bundle. The two sensitivity definitions give 640/611/544/537 (V0, the
original set) and 635/606/541/534 (V1, error corrections only). See section 5a.

Gate A is the plan's ladder, reachable only under a strict integer cast that nulls the 78
patients whose `AGE` is the literal token `'90+'`. It is a RECONCILIATION target reproduced
by `02_build_cohort.py --replicate-plan`, never a result.
Gate B is the corrected ladder and is PRIMARY (`CLAUDE.md` section 7.2): the curated `main`
view ships those 78 patients as `AGE = 90`, and they satisfy the plan's own `age >= 18`.

The hematoma column is shared because it is IDENTICAL on both ladders at every step: the 78
restored nonagenarians contribute ZERO hematoma events. Measured, not assumed. See
`COHORT_LADDER_EVENTS` below.

Three operational details are NOT in the plan prose but are required to hit those counts;
they are recorded in `derived/notes_cpt.md` sections 4 and 5 for `02_build_cohort.py`.

===============================================================================
2. WHY PREFIX MATCHING, AND WHY THE SEROMA GUARD IS NOT OPTIONAL
===============================================================================
NSQIP reoperation / readmission ICD-10 fields carry the code at whatever specificity the
abstractor entered. All of these literal forms are present in the 2012-2024 PUF:

    M96.83 (30)   M96.830 (638)   M96.831 (244)          <- 4-char truncations are real
    M96.84 (189)  M96.840 (1,668) M96.841 (926)
    L76.3  (1,013) L76.31 (347)   L76.32 (6,815)
    G97.6  (54)   G97.61 (950)    G97.62 (193)  G97.63 (219)  G97.64 (32)
    G97.5  (bare) G97.51          G97.52

An exact-code set would silently drop every truncated form, so matching is by PREFIX.
But the same families also contain SEROMA members, which the plan excludes:

    M96.842 (606)  M96.843 (209)   <- swallowed by prefix M9684 unless guarded
    L76.33  (35)   L76.34 (1,494)  <- swallowed by prefix L763  unless guarded
    G97.63  (219)  G97.64 (32)     <- swallowed by prefix G976  unless guarded

Hence the matcher is "prefix match AND NOT in the seroma exclusion set", implemented in
`is_hematoma_icd10`. The guard is load-bearing, not decorative. Re-measured 2026-07-30 at
the post-exclusion step under the V0 code set, and identical on both ladders (113,411
Gate A / 113,469 Gate B):

    with both seroma guards (the V0 definition)       544
    ICD-10 seroma guard dropped                       561   (+17)
    ICD-9 998.13 guard dropped as well                562   (+1)

562 is the figure `CLAUDE.md` 6d.5 quotes and it means BOTH guards dropped. Dropping only
the ICD-10 guard gives 561; the extra case is a slot-1 `REOPORICD91 = '998.13'`.

**G97.63 and G97.64 were missing from that guard until 2026-07-31.** They are the
postprocedural SEROMA members of the G97.6x family, verified against the CMS FY2026
tabular order file, and 1 of the 537 V0 events rested solely on them. They are in the
primary guard from V1 onward, which enforces the plan's stated intent rather than
changing it. See section 5a.

===============================================================================
3. THE HEMATOMA UNION IS THREE CODE SYSTEMS OVER THE REOPERATION-1 SLOT
===============================================================================
Reoperation-1 fields only. Measured arm contributions at the ANALYTIC step (112,585 Gate A
/ 112,642 Gate B), identical on both ladders. The arms overlap, so they do not add to the
union:

    definition   CPT arm   ICD-9 arm   ICD-10 arm   union
    V0 (old)         419         102          271     537
    V1               419         102          264     534
    V2 (PRIMARY)     419         102          277     541

The CPT and ICD-9 arms are the same in all three variants; only the ICD-10 arm moves.
Adding the reoperation-2 slot raises the V0 identified-cohort count to 648 (elective 617)
on either ladder and does NOT reproduce the plan's 640.

    REOPORCPT1     unplanned reoperation 1 CPT     2012-2024   -> HEMATOMA_REOP_CPTS
    REOPORICD91    unplanned reoperation 1 ICD-9   2012-2020   -> HEMATOMA_ICD9
    REOPOR1ICD101  unplanned reoperation 1 ICD-10  2014-2024   -> HEMATOMA_ICD10_*

Note the asymmetric spelling: the ICD-9 column is `REOPORICD91` with NO "1" after REOPOR,
while the ICD-10 column is `REOPOR1ICD101` WITH one. `REOPOR1ICD91` does not exist.
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, Iterable, Tuple

# ---------------------------------------------------------------------------
# Normalization. Every comparison in this module runs on normalized values.
# ---------------------------------------------------------------------------

# Tokens that mean "no code recorded". NSQIP writes several of these; pandas and numpy
# add the rest on the CSV round trip performed by `01_export_cohort.py`.
_NULL_TOKENS: FrozenSet[str] = frozenset(
    {"", ".", "-", "NA", "N/A", "NAN", "NAT", "NONE", "NULL", "<NA>", "-99", "-99.0"}
)


def _to_text(value: Any) -> str:
    """Coerce any scalar to a stripped string, mapping every null form to ''.

    Handles None, float('nan'), numpy.float64 nan, pandas NA / NaT (via their repr),
    the NSQIP '-99' sentinel, and stray whitespace. Never raises.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        text = str(value)
    except Exception:  # pragma: no cover - a __str__ that raises is not our problem
        return ""
    text = "".join(text.split())  # drops leading, trailing AND internal whitespace
    if text.upper() in _NULL_TOKENS:
        return ""
    return text


def normalize_cpt(code: Any) -> str:
    """Normalize a CPT code to its bare 5-character uppercase form.

    CPT arrives as a string in the parquet layer, but survives a CSV round trip as a
    float-like string when a column is read without `dtype=str`. Both are handled:

        normalize_cpt('22551')    -> '22551'
        normalize_cpt(' 22551 ')  -> '22551'
        normalize_cpt('22551.0')  -> '22551'
        normalize_cpt(22551.0)    -> '22551'
        normalize_cpt(22551)      -> '22551'
        normalize_cpt(None)       -> ''
        normalize_cpt(float('nan')) -> ''

    Uppercasing preserves Category II / III codes, which carry a trailing letter.
    """
    text = _to_text(code).upper()
    if not text:
        return ""
    # Strip a pure-zero fractional part, e.g. '22551.0' / '22551.00' -> '22551'.
    if "." in text:
        head, _, tail = text.partition(".")
        if head.isdigit() and (tail == "" or set(tail) == {"0"}):
            text = head
    return text


def normalize_icd(code: Any) -> str:
    """Normalize an ICD-9 or ICD-10 code: strip whitespace, drop periods, uppercase.

        normalize_icd('M96.83')  -> 'M9683'
        normalize_icd('m9683 ')  -> 'M9683'
        normalize_icd('M9683')   -> 'M9683'
        normalize_icd('998.12')  -> '99812'
        normalize_icd(None)      -> ''

    Lowercase codes are genuinely present in the PUF (k91.89, t81.49xa, z98.82 and
    others), so the uppercase step is load-bearing, not defensive.
    """
    return _to_text(code).replace(".", "").upper()


# ---------------------------------------------------------------------------
# 1. Principal procedure codes: cohort inclusion
# ---------------------------------------------------------------------------
# Plan, "Inclusion": age >= 18, principal CPT in this set, elective. The principal `CPT`
# column ONLY; `OTHERCPT*` and `CONCPT*` are never consulted for cohort entry.

PRINCIPAL_CPTS: FrozenSet[str] = frozenset({"22551", "22554", "63075"})

# ---------------------------------------------------------------------------
# 2. Exclusion codes, scanned across OTHERCPT1..OTHERCPT10
# ---------------------------------------------------------------------------
# Plan, "Exclusion". Kept as three named subsets so Figure 1 can report the reason a case
# left the cohort, then unioned into EXCLUSION_CPTS for the actual filter.

EXCLUSION_CORPECTOMY_CPTS: FrozenSet[str] = frozenset({"63081", "63082", "22548"})
EXCLUSION_POSTERIOR_CERVICAL_CPTS: FrozenSet[str] = frozenset(
    {"22600", "22842", "22843", "22844", "63001", "63015", "63045", "63047"}
)
EXCLUSION_ARTHROPLASTY_CPTS: FrozenSet[str] = frozenset({"22856"})

EXCLUSION_CPTS: FrozenSet[str] = (
    EXCLUSION_CORPECTOMY_CPTS
    | EXCLUSION_POSTERIOR_CERVICAL_CPTS
    | EXCLUSION_ARTHROPLASTY_CPTS
)

# Reason label per excluded code, for the Figure 1 STROBE flow.
#
# CLAUDE.md 6d.6 and 6j.2, binding: the middle set MUST be described as "concurrent
# posterior spinal decompression or instrumentation" and NEVER as "posterior cervical".
# 63047 is a lumbar code and 22842 to 22844 are not cervical-specific, so a spine reviewer
# reads "posterior cervical procedure" as an error. The CPT set itself is honored verbatim
# because it is what produces the 113,469 rung; only the description is corrected. An
# earlier revision of this file carried the banned wording and commented it as safe to
# render; both the label and that comment are now gone.
EXCLUSION_REASON: Dict[str, str] = {
    **{c: "Concurrent corpectomy" for c in EXCLUSION_CORPECTOMY_CPTS},
    **{c: "Concurrent posterior spinal decompression or instrumentation"
       for c in EXCLUSION_POSTERIOR_CERVICAL_CPTS},
    **{c: "Cervical disc arthroplasty" for c in EXCLUSION_ARTHROPLASTY_CPTS},
}

# Display order for the Figure 1 exclusion boxes. The SEQUENCE is unchanged (corpectomy,
# posterior set, arthroplasty); only the middle label tracks the 6d.6 correction above,
# which the import-time invariant below requires.
EXCLUSION_REASON_ORDER: Tuple[str, ...] = (
    "Concurrent corpectomy",
    "Concurrent posterior spinal decompression or instrumentation",
    "Cervical disc arthroplasty",
)

# ---------------------------------------------------------------------------
# 3. Fusion levels
# ---------------------------------------------------------------------------
# Plan, "Exposures": fusion levels 1 / 2 / >=3, derived by counting repeated CPT 22552
# across OTHERCPT1..10. 22552 is the "each additional interspace" add-on to 22551, so one
# occurrence means a second level, two occurrences a third, and so on.
#
# Measured distribution of 22552 occurrences over OTHERCPT1..10, on the CORRECTED IDENTIFIED
# cohort (Gate B, n = 124,344; the counts below sum to 124,344, not to the plan's 124,266):
#   0 -> 70,120 | 1 -> 47,262 | 2 -> 5,713 | 3 -> 1,126 | 4 -> 102 | 5..8 -> 21
# The same tally on the Gate A identified cohort (n = 124,266) is
#   0 -> 70,064 | 1 -> 47,243 | 2 -> 5,711 | 3 -> 1,125 | 4 -> 102 | 5..8 -> 21

ADDITIONAL_LEVEL_CPT: str = "22552"

# Stratum tokens. Plain, non-snake_case; `variable_dict.pretty_level()` owns display text.
LEVEL_STRATUM_ORDER: Tuple[str, str, str] = ("1", "2", ">=3")
LEVEL_STRATUM_UNKNOWN: str = ""

# ---------------------------------------------------------------------------
# 4. Anterior instrumentation covariate, and the airway-catastrophe marker
# ---------------------------------------------------------------------------
# Plan, "Covariates": anterior instrumentation (22845-22847), a plate-fixation covariate,
# scanned across OTHERCPT1..10. Measured: 71,946 of the 113,469-case Gate B exclusion step
# (71,919 of the 113,411-case Gate A step).
#
# Plan, "Secondary outcomes": reoperative tracheostomy, CPT 31600 in REOPORCPT1. This marker
# is STEP-SPECIFIC and the plan's "61" is an identified-cohort figure. Measured 2026-07-30,
# slot 1 (REOPORCPT1) only, and IDENTICAL on both ladders:
#   identified 61 | elective 46 | post-exclusion 32 | analytic 24
# Slots 1+2 (REOPORCPT1 or REOPOR2CPT1): 73 | 55 | 38 | 30.
# `CLAUDE.md` 6c fixes slot 1 as the convention; slots 1+2 belong in the ascertainment
# supplement. Re-measure in the step that owns the number; do not copy it from here.

ANTERIOR_INSTRUMENTATION_CPTS: FrozenSet[str] = frozenset({"22845", "22846", "22847"})
TRACHEOSTOMY_CPT: str = "31600"

# ---------------------------------------------------------------------------
# 5. Hematoma ascertainment: a UNION of three code systems
# ---------------------------------------------------------------------------
# Plan, "Primary outcome". A case is a hematoma event if ANY arm fires.

# --- arm 1: reoperation CPT (REOPORCPT1), all 13 years -----------------------
HEMATOMA_REOP_CPTS: FrozenSet[str] = frozenset({"10140", "21501", "35800", "20005"})

# --- arm 2: reoperation ICD-9 (REOPORICD91), 2012-2020 ----------------------
# Exact match. The 4-digit truncation '998.1' is present (282 rows PUF-wide) but is
# ambiguous between 998.11 / 998.12 / 998.13 and is therefore NOT counted.
HEMATOMA_ICD9: FrozenSet[str] = frozenset({normalize_icd("998.11"), normalize_icd("998.12")})
SEROMA_ICD9_EXCLUDED: FrozenSet[str] = frozenset({normalize_icd("998.13")})

# --- arm 3: reoperation ICD-10 (REOPOR1ICD101), 2014-2024 -------------------
#
# ###########################################################################
# THE PRIMARY OUTCOME DEFINITION SWITCH. Change ONE line to change the study's
# primary outcome; everything downstream follows. See section 5a below and
# `derived/notes_cpt.md` section 10.
# ###########################################################################
HEMATOMA_DEFINITION: str = "V2"

# Prefix families. Stored normalized (no periods, uppercase) because that is the space
# every comparison happens in. No entry is a prefix of another, which is asserted below
# for every variant; that is what guarantees the family cannot be double-counted.
#
# Descriptor comments below are the FY2026 CMS official long titles, verified 2026-07-31
# against `icd10cm_order_2026.txt` and cross-checked FY2016 through FY2026. Where the
# title changed at FY2017 both are given, because this cohort spans that boundary.
_ICD10_V0: Dict[str, Any] = dict(
    prefixes=(
        normalize_icd("M96.83"),    # postprocedural hemorrhage, musculoskeletal structure
        normalize_icd("M96.84"),    # postprocedural hematoma and seroma, musculoskeletal
        normalize_icd("L76.2"),     # postprocedural hemorrhage, skin and subcutaneous
        normalize_icd("L76.3"),     # postprocedural hematoma and seroma, skin and subcut.
        normalize_icd("G97.6"),     # postprocedural hematoma and seroma, nervous system
        normalize_icd("T81.0"),     # DOES NOT EXIST IN ICD-10-CM. See section 5a.
        normalize_icd("T79.2XXA"),  # traumatic secondary/recurrent hemorrhage and seroma
    ),
    exact=frozenset({normalize_icd("M96.810"), normalize_icd("M96.820")}),
    seroma=frozenset({
        normalize_icd("M96.842"),
        normalize_icd("M96.843"),
        normalize_icd("L76.33"),
        normalize_icd("L76.34"),
    }),
    label="Original code set, retained as a sensitivity analysis",
)

# V1 = error corrections only. T81.0 removed (no such ICD-10-CM code in any fiscal year);
# M96.820 removed (accidental puncture and laceration, not a bleeding code); G97.63 and
# G97.64 added to the seroma guard (they are the postprocedural SEROMA members of the
# G97.6x family and the plan already excludes seroma-only codes).
_ICD10_V1: Dict[str, Any] = dict(
    prefixes=tuple(p for p in _ICD10_V0["prefixes"] if p != normalize_icd("T81.0")),
    exact=frozenset({normalize_icd("M96.810")}),
    seroma=_ICD10_V0["seroma"] | frozenset({normalize_icd("G97.63"), normalize_icd("G97.64")}),
    label="Error-corrected code set, hematoma arm only",
)

# V2 = V1 plus the G97.5x postprocedural-HEMORRHAGE arm of the nervous-system family.
# PRIMARY. Including G97.6 (hematoma) while omitting G97.5 (hemorrhage) is internally
# inconsistent: after an ACDF, a postprocedural hemorrhage of a nervous system organ
# requiring reoperation is the same clinical event as a postprocedural hematoma. The
# omission is also era-dependent, which is the decisive argument: in FY2016 ICD-10-CM,
# G97.51/G97.52 were titled "Postprocedural hemorrhage AND HEMATOMA of a nervous system
# organ or structure" and the G97.6x hematoma codes DID NOT EXIST. FY2017 split them.
# Measured in this cohort: G97.5x occurs in 2015-2016 (10 of 14 occurrences) and G97.6x
# does not appear before 2017. Omitting G97.5x therefore drops nervous-system hematomas
# systematically from the earliest ICD-10 years of the study window.
_ICD10_V2: Dict[str, Any] = dict(
    prefixes=_ICD10_V1["prefixes"] + (normalize_icd("G97.5"),),
    exact=_ICD10_V1["exact"],
    seroma=_ICD10_V1["seroma"],
    label="Error-corrected code set plus the nervous-system hemorrhage arm",
)

HEMATOMA_ICD10_VARIANTS: Dict[str, Dict[str, Any]] = {
    "V0": _ICD10_V0, "V1": _ICD10_V1, "V2": _ICD10_V2,
}
HEMATOMA_DEFINITION_ORDER: Tuple[str, ...] = ("V0", "V1", "V2")

# The selected variant, exported under the historical names so every downstream import
# keeps working and lands on the PRIMARY definition by default.
HEMATOMA_ICD10_PREFIXES: Tuple[str, ...] = \
    HEMATOMA_ICD10_VARIANTS[HEMATOMA_DEFINITION]["prefixes"]
HEMATOMA_ICD10_EXACT: FrozenSet[str] = HEMATOMA_ICD10_VARIANTS[HEMATOMA_DEFINITION]["exact"]
# Seroma-only members that the prefix families would otherwise swallow. Checked FIRST in
# `is_hematoma_icd10`, before both the exact set and the prefix scan.
SEROMA_ICD10_EXCLUDED: FrozenSet[str] = \
    HEMATOMA_ICD10_VARIANTS[HEMATOMA_DEFINITION]["seroma"]

# ---------------------------------------------------------------------------
# 5a. WHY THE CODE SET CHANGED, AND WHAT EACH VARIANT MEASURES
# ---------------------------------------------------------------------------
# Every descriptor claim below was verified 2026-07-31 against the CMS ICD-10-CM
# code-description order files for FY2016 through FY2026 (the FY2016 file from the
# CDC/NCHS FTP archive, FY2018 and FY2021-FY2026 from cms.gov, FY2017 / FY2019 / FY2020
# from CDC/NCHS). Nothing here is transcribed from a secondary lookup site.
#
#   T81.0    DOES NOT EXIST in ICD-10-CM in ANY fiscal year FY2016-FY2026. The descriptor
#            "Hemorrhage and hematoma complicating a procedure" is WHO ICD-10, not
#            ICD-10-CM; ICD-10-CM has no T81.0 subcategory at all (T81 goes .1x, .3x,
#            .4x, .5x, .6, .7x, .8x, .9x). It matched 0 rows in 13 years of PUF. REMOVED.
#   M96.820  "Accidental puncture and laceration of a musculoskeletal structure during a
#            musculoskeletal system procedure." Valid and billable in every fiscal year,
#            but NOT a bleeding code. 2 of the 537 events under V0 rest solely on it.
#            REMOVED, which also resolves the open UNVERIFIED_DESCRIPTORS item that
#            `CLAUDE.md` 6d.7 and `derived/notes_cpt.md` 6.3 left blocking Supplement S1.
#   M96.810  KEPT. "Intraoperative hemorrhage and hematoma of a musculoskeletal structure
#            complicating a musculoskeletal system procedure." It is a bleeding code.
#            It matches 0 rows in this cohort, so it is inert either way; it stays because
#            it is correct, not because it does anything.
#   G97.63   "Postprocedural seroma of a nervous system organ or structure following a
#   G97.64   nervous system procedure" / "... following other procedure". These are SEROMA
#            codes swept in by the G97.6 prefix. 1 of the 537 events under V0 rests solely
#            on them. ADDED TO THE SEROMA GUARD, which enforces the plan's own stated
#            intent to exclude seroma-only codes rather than changing it.
#   G97.65   DO NOT EXIST in any fiscal year. An earlier revision of this module named them
#   G97.66   as "the seroma members of the G97.6x family" in a constant called
#            SEROMA_ICD10_EXCLUDED_EXTENDED. That was wrong: the seroma members are
#            G97.63 and G97.64, which are now in the primary guard. The constant is GONE.
#   G97.51   "Postprocedural hemorrhage of a nervous system organ or structure following a
#   G97.52   nervous system procedure" / "... following other procedure" from FY2017.
#            In FY2016 both were titled "Postprocedural hemorrhage AND HEMATOMA of a
#            nervous system organ or structure ...", and G97.6x did not exist. ADDED in V2.
#            7 cases in the analytic cohort carry G97.51 and are counted by no other arm.
#
# The three variants, measured on the Gate B analytic cohort (n = 112,642) and identical
# on the Gate A cohort (n = 112,585). See COHORT_LADDER_EVENTS_BY_DEFINITION below.
#
#   V0  537 events   the original code set, kept as a pre-specified sensitivity
#   V1  534 events   error corrections only (-2 M96.820-only, -1 G97.63-only)
#   V2  541 events   V1 plus the G97.5x hemorrhage arm (+7 G97.51-only)   <- PRIMARY
#
# `03_analysis.py` publishes V0 and V1 as named rows of the Supplement S2 sensitivity
# suite, so the effect of the definition change is on the page rather than in a note.

# ---------------------------------------------------------------------------
# 6. Dysphagia, ICD-10 only, 2014-2024
# ---------------------------------------------------------------------------
# Plan, section 4. The family is R13.1 and everything beneath it. A SINGLE prefix is used,
# so `is_dysphagia_icd10` is one boolean per code and per-subcode counts partition the
# family exactly once. R13.1 is itself a prefix of R13.10 / R13.12 / ..., which is why the
# family total must be counted as "cases where the predicate is True", NEVER as the sum of
# the individual sub-code counts. Summing would count an R13.10 record twice.
#
# R13 (unspecified symptoms) and R13.0 (aphagia) are outside the family and do not match.
DYSPHAGIA_ICD10_PREFIXES: Tuple[str, ...] = (normalize_icd("R13.1"),)

# The six literals the plan enumerates. The prefix family and this exact set give IDENTICAL
# counts at every ladder step because R13.11 has zero occurrences in this cohort. The counts
# are step-specific; the pair below is the POST-EXCLUSION step (113,411 Gate A / 113,469
# Gate B), which is where they were first measured:
#   readmission (READMRELICD101) 246 Gate A / 247 Gate B ; reoperation (REOPOR1ICD101) 43 both
# At the ANALYTIC step they are 244 / 41 (Gate A) and 245 / 41 (Gate B).
# Kept so a strict-literal sensitivity analysis is available without retyping a code.
DYSPHAGIA_ICD10_PLAN_CODES: FrozenSet[str] = frozenset(
    normalize_icd(c) for c in ("R13.10", "R13.1", "R13.12", "R13.19", "R13.13", "R13.14")
)

# One bucket per distinct normalized code, for the Table 5 subtype breakdown. Every key is
# a member of the prefix family, so the buckets partition it.
# R13.1 carries the ICD-10-CM CATEGORY HEADER title, "Dysphagia" (verified 2026-07-30
# against the AAPC and icd10data ICD-10-CM code lookups: R13.1 is a non-billable header over
# the billable R13.10-R13.19 subcodes, under R13 "Aphagia and dysphagia"). "unspecified"
# belongs to R13.10 and must not be attached to R13.1 (CLAUDE.md 6d.7: no invented descriptor).
DYSPHAGIA_ICD10_SUBTYPE: Dict[str, str] = {
    normalize_icd("R13.1"): "Dysphagia",
    normalize_icd("R13.10"): "Dysphagia, unspecified",
    normalize_icd("R13.11"): "Dysphagia, oral phase",
    normalize_icd("R13.12"): "Dysphagia, oropharyngeal phase",
    normalize_icd("R13.13"): "Dysphagia, pharyngeal phase",
    normalize_icd("R13.14"): "Dysphagia, pharyngoesophageal phase",
    normalize_icd("R13.19"): "Other dysphagia",
}

# ---------------------------------------------------------------------------
# 7. Human-readable descriptors, for Supplement S1 (code definitions)
# ---------------------------------------------------------------------------
# Free of snake_case and EM-dashes so they can be written straight into a table.
# Any code whose official descriptor could not be verified is listed in
# UNVERIFIED_DESCRIPTORS instead of being given a guessed label.

CODE_LABEL: Dict[str, str] = {
    # principal
    "22551": "Anterior cervical discectomy with interbody arthrodesis, single interspace",
    "22554": "Anterior cervical interbody arthrodesis, single interspace",
    "63075": "Anterior cervical discectomy with decompression, single interspace",
    "22552": "Anterior cervical interbody arthrodesis, each additional interspace",
    # exclusions
    "63081": "Anterior cervical corpectomy with decompression, single segment",
    "63082": "Anterior cervical corpectomy with decompression, each additional segment",
    "22548": "Anterior transoral or extraoral arthrodesis, clivus to C2",
    "22600": "Posterior or posterolateral cervical arthrodesis, single level",
    "22842": "Posterior segmental instrumentation, 3 to 6 vertebral segments",
    "22843": "Posterior segmental instrumentation, 7 to 12 vertebral segments",
    "22844": "Posterior segmental instrumentation, 13 or more vertebral segments",
    "63001": "Cervical laminectomy with decompression, 1 or 2 vertebral segments",
    "63015": "Cervical laminectomy with decompression, more than 2 vertebral segments",
    "63045": "Cervical laminectomy, facetectomy and foraminotomy, single segment",
    "63047": "Lumbar laminectomy, facetectomy and foraminotomy, single segment",
    "22856": "Cervical total disc arthroplasty, anterior approach, single interspace",
    # instrumentation and airway marker
    "22845": "Anterior instrumentation, 2 to 3 vertebral segments",
    "22846": "Anterior instrumentation, 4 to 7 vertebral segments",
    "22847": "Anterior instrumentation, 8 or more vertebral segments",
    "31600": "Tracheostomy, planned",
    # hematoma reoperation CPT
    "10140": "Incision and drainage of hematoma, seroma or fluid collection",
    "21501": "Incision and drainage of deep abscess or hematoma, soft tissues of the neck",
    "35800": "Exploration for postoperative hemorrhage, thrombosis or infection, neck",
    "20005": "Incision and drainage of soft tissue abscess, deep or complicated",
    # hematoma ICD-9
    "99811": "Hemorrhage complicating a procedure",
    "99812": "Hematoma complicating a procedure",
    "99813": "Seroma complicating a procedure",
    # hematoma ICD-10 prefix families.
    # Every string below is the CMS FY2026 official long title, verified 2026-07-31 against
    # `icd10cm_order_2026.txt` and cross-checked across FY2016-FY2026. Five of them were
    # WRONG in an earlier revision of this module while being published with "Descriptor
    # verified: Yes"; the corrections are noted inline.
    "M9683": "Postprocedural hemorrhage of a musculoskeletal structure following a procedure",
    # was "Postprocedural hematoma of ...": the official title carries seroma as well.
    "M9684": "Postprocedural hematoma and seroma of a musculoskeletal structure following "
             "a procedure",
    "L762": "Postprocedural hemorrhage of skin and subcutaneous tissue following a procedure",
    # was "Postprocedural hematoma of ...": the official title carries seroma as well.
    "L763": "Postprocedural hematoma and seroma of skin and subcutaneous tissue following "
            "a procedure",
    # was "Postprocedural hemorrhage or hematoma of a nervous system structure". G97.6 is
    # hematoma AND SEROMA and carries no hemorrhage; hemorrhage is G97.5-.
    "G976": "Postprocedural hematoma and seroma of a nervous system organ or structure "
            "following a procedure",
    "G975": "Postprocedural hemorrhage of a nervous system organ or structure following "
            "a procedure",
    "G9751": "Postprocedural hemorrhage of a nervous system organ or structure following "
             "a nervous system procedure",
    "G9752": "Postprocedural hemorrhage of a nervous system organ or structure following "
             "other procedure",
    # was "Traumatic secondary and recurrent hemorrhage, initial encounter": the official
    # title carries seroma as well, and T79.2 also carries an Excludes2 for complications
    # following a medical procedure, which is why it contributes almost nothing here.
    "T792XXA": "Traumatic secondary and recurrent hemorrhage and seroma, initial encounter",
    "M96810": "Intraoperative hemorrhage and hematoma of a musculoskeletal structure "
              "complicating a musculoskeletal system procedure",
    # Removed from the code set at V1. The descriptor is now VERIFIED and kept so that
    # Supplement S1 can say plainly what was dropped and why.
    "M96820": "Accidental puncture and laceration of a musculoskeletal structure during "
              "a musculoskeletal system procedure",
    # seroma exclusions
    "M96842": "Postprocedural seroma of a musculoskeletal structure following a "
              "musculoskeletal system procedure",
    "M96843": "Postprocedural seroma of a musculoskeletal structure following other procedure",
    "L7633": "Postprocedural seroma of skin and subcutaneous tissue following a "
             "dermatologic procedure",
    "L7634": "Postprocedural seroma of skin and subcutaneous tissue following other procedure",
    "G9763": "Postprocedural seroma of a nervous system organ or structure following a "
             "nervous system procedure",
    "G9764": "Postprocedural seroma of a nervous system organ or structure following "
             "other procedure",
    **DYSPHAGIA_ICD10_SUBTYPE,
}

# ---------------------------------------------------------------------------
# 7a. Readmission-cause descriptors for Table 5C and Supplement S3
# ---------------------------------------------------------------------------
# The 25 most frequent `READMRELICD101` values in this cohort. Table 5C previously
# published these as BARE CODES in a column headed "Readmission cause", which is unreadable
# to a clinical reviewer. Every descriptor below is the CMS FY2026 official long title,
# verified 2026-07-31 against `icd10cm_order_2026.txt`; nothing is paraphrased and nothing
# is invented (CLAUDE.md 6d.7). A code outside this list still renders as a bare code, and
# `04_dysphagia.py` says so in the caption rather than guessing.
#
# `M96.842` and `L76.34` are also in this top-25 list but are NOT repeated here: they
# already carry an identical descriptor in CODE_LABEL as seroma-guard members, and an
# import-time assertion forbids one code having two sources of truth for its label.
#
# Note `R13.1` is the non-billable ICD-10-CM CATEGORY HEADER, titled simply "Dysphagia";
# "unspecified" belongs to `R13.10`. It is already in DYSPHAGIA_ICD10_SUBTYPE above.
READMISSION_CODE_LABEL: Dict[str, str] = {
    normalize_icd("G89.18"): "Other acute postprocedural pain",
    normalize_icd("M48.02"): "Spinal stenosis, cervical region",
    normalize_icd("M54.2"): "Cervicalgia",
    normalize_icd("L76.32"): "Postprocedural hematoma of skin and subcutaneous tissue "
                             "following other procedure",
    normalize_icd("R22.1"): "Localized swelling, mass and lump, neck",
    normalize_icd("R55"): "Syncope and collapse",
    normalize_icd("M96.840"): "Postprocedural hematoma of a musculoskeletal structure "
                              "following a musculoskeletal system procedure",
    normalize_icd("R53.1"): "Weakness",
    normalize_icd("M47.12"): "Other spondylosis with myelopathy, cervical region",
    normalize_icd("M54.12"): "Radiculopathy, cervical region",
    normalize_icd("R41.82"): "Altered mental status, unspecified",
    normalize_icd("G97.61"): "Postprocedural hematoma of a nervous system organ or "
                             "structure following a nervous system procedure",
    normalize_icd("R07.9"): "Chest pain, unspecified",
    normalize_icd("E87.1"): "Hypo-osmolality and hyponatremia",
    normalize_icd("S10.93XA"): "Contusion of unspecified part of neck, initial encounter",
    normalize_icd("T81.89XA"): "Other complications of procedures, not elsewhere "
                               "classified, initial encounter",
    normalize_icd("K91.89"): "Other postprocedural complications and disorders of "
                             "digestive system",
    normalize_icd("N17.9"): "Acute kidney failure, unspecified",
    normalize_icd("I48.91"): "Unspecified atrial fibrillation",
}

# Empty since 2026-07-31. `M96.820` sat here because nobody had opened the tabular list;
# it has now been read (it is accidental puncture and laceration, not a bleeding code) and
# the code was removed from the outcome definition rather than labelled into it. The set
# stays as the mechanism: any future code whose official descriptor cannot be verified goes
# here and `code_label` returns '' for it (CLAUDE.md 6d.7 - never invent a descriptor).
UNVERIFIED_DESCRIPTORS: FrozenSet[str] = frozenset()

# Codes VERIFIED NOT TO EXIST in ICD-10-CM, in any fiscal year FY2016-FY2026. These have no
# descriptor and never will, which is a DIFFERENT state from UNVERIFIED_DESCRIPTORS above:
# nothing is outstanding and Supplement S1 is not blocked by them. They are kept so the
# supplement can say plainly what the original code set carried and why it was dropped.
NONEXISTENT_ICD10: FrozenSet[str] = frozenset({normalize_icd("T81.0")})

# ---------------------------------------------------------------------------
# 8. Column contract: where each code set is scanned
# ---------------------------------------------------------------------------
# Column names only, no logic. `02_build_cohort.py` should reference these instead of
# rebuilding the f-string ranges, so a warehouse rename breaks in exactly one place.

PRINCIPAL_CPT_COLUMN: str = "CPT"
OTHERCPT_COLUMNS: Tuple[str, ...] = tuple(f"OTHERCPT{i}" for i in range(1, 11))
CONCPT_COLUMNS: Tuple[str, ...] = tuple(f"CONCPT{i}" for i in range(1, 11))

HEMATOMA_CPT_COLUMN: str = "REOPORCPT1"
HEMATOMA_ICD9_COLUMN: str = "REOPORICD91"      # note: NOT 'REOPOR1ICD91', which does not exist
HEMATOMA_ICD10_COLUMN: str = "REOPOR1ICD101"

DYSPHAGIA_READMISSION_ICD10_COLUMN: str = "READMRELICD101"
DYSPHAGIA_REOPERATION_ICD10_COLUMN: str = "REOPOR1ICD101"

# ---------------------------------------------------------------------------
# 9. Verified cohort ladders, machine-readable. TWO of them.
# ---------------------------------------------------------------------------
# So `02_build_cohort.py` asserts against these instead of carrying its own copies.
# Both re-measured against the parquet on 2026-07-30 (Batch 1 review).
#
# `CLAUDE.md` section 7 defines two gates and this module exports one dict per gate:
#
#   COHORT_LADDER_N_PLAN  Gate A, RECONCILIATION ONLY. The plan's ladder, reachable only
#                         under the strict integer cast that nulls AGE = '90+' and silently
#                         drops 78 eligible nonagenarians. `02 --replicate-plan` must
#                         reproduce it EXACTLY. It goes in the ascertainment supplement.
#                         It is NOT a result and must never be quoted as one.
#
#   COHORT_LADDER_N       Gate B, PRIMARY. The corrected ladder. Every table, figure, model
#                         and manuscript sentence uses this one. The curated `main` view
#                         ships AGE as SMALLINT with '90+' normalized to 90, so the export
#                         bundle CONTAINS those 78 patients and reproducing the plan's
#                         numbers would mean deliberately re-introducing the cast bug.
#
# The unqualified name is the primary cohort on purpose: an import that forgets the
# distinction lands on the ladder the manuscript actually uses.
#
# There is exactly ONE event dict because the two ladders have IDENTICAL event counts at
# every step. Measured, not assumed: the 78 restored nonagenarians contribute ZERO hematoma
# events (640 / 611 / 544 / 537 on both). That is also why Gate C tolerances are tight: a
# 57-patient shift that carries no events cannot move an odds ratio materially.

COHORT_LADDER_N_PLAN: Dict[str, int] = {
    "identified": 124266,
    "elective": 121142,
    "exclusions": 113411,
    "analytic": 112585,
}
COHORT_LADDER_N: Dict[str, int] = {
    "identified": 124344,
    "elective": 121210,
    "exclusions": 113469,
    "analytic": 112642,
}
# RE-PINNED 2026-07-31 to the V2 primary outcome definition (section 5a). Every figure
# below was MEASURED against the export bundle, not computed by arithmetic from the old
# 640 / 611 / 544 / 537. The event ladder is still shared by Gate A and Gate B: the 78
# restored nonagenarians contribute zero hematoma events under all three variants.
COHORT_LADDER_EVENTS_BY_DEFINITION: Dict[str, Dict[str, int]] = {
    "V0": {"identified": 640, "elective": 611, "exclusions": 544, "analytic": 537},
    "V1": {"identified": 635, "elective": 606, "exclusions": 541, "analytic": 534},
    "V2": {"identified": 642, "elective": 613, "exclusions": 548, "analytic": 541},
}
COHORT_LADDER_EVENTS: Dict[str, int] = \
    dict(COHORT_LADDER_EVENTS_BY_DEFINITION[HEMATOMA_DEFINITION])
COHORT_LADDER_ORDER: Tuple[str, ...] = ("identified", "elective", "exclusions", "analytic")

# The Gate B minus Gate A delta at every step, i.e. the nonagenarians the strict cast drops.
COHORT_LADDER_NONAGENARIAN_DELTA: Dict[str, int] = {
    step: COHORT_LADDER_N[step] - COHORT_LADDER_N_PLAN[step] for step in COHORT_LADDER_ORDER
}

# ---------------------------------------------------------------------------
# 10. Integrity assertions. A bad import fails loudly and immediately.
# ---------------------------------------------------------------------------

assert len(PRINCIPAL_CPTS) == 3, f"expected 3 principal ACDF CPTs, got {len(PRINCIPAL_CPTS)}"
assert len(EXCLUSION_CPTS) == 12, f"expected 12 exclusion CPTs, got {len(EXCLUSION_CPTS)}"
assert len(HEMATOMA_REOP_CPTS) == 4
assert len(HEMATOMA_ICD9) == 2
assert HEMATOMA_DEFINITION in HEMATOMA_ICD10_VARIANTS, (
    f"HEMATOMA_DEFINITION {HEMATOMA_DEFINITION!r} is not one of "
    f"{sorted(HEMATOMA_ICD10_VARIANTS)}"
)
assert set(HEMATOMA_ICD10_VARIANTS) == set(HEMATOMA_DEFINITION_ORDER)
assert HEMATOMA_DEFINITION == "V2", (
    "the primary outcome definition is V2 (CLAUDE.md 6d.9). Switching it is a deliberate "
    "act: change this assertion and COHORT_LADDER_EVENTS follows automatically, but every "
    "pinned gate in verify_numbers.py must be re-measured, never recomputed."
)
_VARIANT_SHAPE = {"V0": (7, 2, 4), "V1": (6, 1, 6), "V2": (7, 1, 6)}
for _v, (_np_, _ne, _ns) in _VARIANT_SHAPE.items():
    _spec = HEMATOMA_ICD10_VARIANTS[_v]
    assert len(_spec["prefixes"]) == _np_, f"{_v} prefix count drifted"
    assert len(_spec["exact"]) == _ne, f"{_v} exact-code count drifted"
    assert len(_spec["seroma"]) == _ns, f"{_v} seroma-guard count drifted"
# The three corrections, asserted as facts rather than left to a comment.
assert normalize_icd("T81.0") in _ICD10_V0["prefixes"], "V0 must still carry T81.0"
assert normalize_icd("T81.0") not in _ICD10_V1["prefixes"] + _ICD10_V2["prefixes"]
assert normalize_icd("M96.820") in _ICD10_V0["exact"]
assert normalize_icd("M96.820") not in (_ICD10_V1["exact"] | _ICD10_V2["exact"])
assert {normalize_icd("G97.63"), normalize_icd("G97.64")} <= _ICD10_V1["seroma"]
assert {normalize_icd("G97.63"), normalize_icd("G97.64")} & _ICD10_V0["seroma"] == frozenset()
assert normalize_icd("G97.5") in _ICD10_V2["prefixes"]
assert normalize_icd("G97.5") not in _ICD10_V0["prefixes"] + _ICD10_V1["prefixes"]
assert "SEROMA_ICD10_EXCLUDED_EXTENDED" not in globals(), (
    "SEROMA_ICD10_EXCLUDED_EXTENDED named G97.65 / G97.66, which exist in no ICD-10-CM "
    "fiscal year. The real G97.6x seroma members are G97.63 / G97.64 and they are in the "
    "primary guard from V1 onward. Do not reintroduce the constant."
)

# Every stored constant must already be in normalized form, so a lookup never needs a
# second pass and a typo cannot hide behind normalization.
for _c in PRINCIPAL_CPTS | EXCLUSION_CPTS | HEMATOMA_REOP_CPTS | ANTERIOR_INSTRUMENTATION_CPTS:
    assert normalize_cpt(_c) == _c, f"CPT constant {_c!r} is not in normalized form"
assert normalize_cpt(ADDITIONAL_LEVEL_CPT) == ADDITIONAL_LEVEL_CPT
assert normalize_cpt(TRACHEOSTOMY_CPT) == TRACHEOSTOMY_CPT
_ALL_ICD_CONSTANTS = (
    HEMATOMA_ICD9
    | SEROMA_ICD9_EXCLUDED
    | DYSPHAGIA_ICD10_PLAN_CODES
    | frozenset(DYSPHAGIA_ICD10_PREFIXES)
)
for _spec in HEMATOMA_ICD10_VARIANTS.values():
    _ALL_ICD_CONSTANTS = (_ALL_ICD_CONSTANTS | _spec["exact"] | _spec["seroma"]
                          | frozenset(_spec["prefixes"]))
for _c in _ALL_ICD_CONSTANTS | frozenset(READMISSION_CODE_LABEL):
    assert normalize_icd(_c) == _c, f"ICD constant {_c!r} is not in normalized form"
# The readmission descriptors are a DISPLAY table only; none of them may silently
# redefine a code that already has a study role.
assert not (frozenset(READMISSION_CODE_LABEL) & frozenset(CODE_LABEL)), (
    "a readmission descriptor shadows a CODE_LABEL entry; keep one source per code"
)

# Mutually exclusive CPT sets must not overlap.
assert not (PRINCIPAL_CPTS & EXCLUSION_CPTS), "principal and exclusion CPT sets overlap"
assert ADDITIONAL_LEVEL_CPT not in PRINCIPAL_CPTS
assert ADDITIONAL_LEVEL_CPT not in EXCLUSION_CPTS
assert not (PRINCIPAL_CPTS & ANTERIOR_INSTRUMENTATION_CPTS)
assert not (EXCLUSION_CPTS & ANTERIOR_INSTRUMENTATION_CPTS), (
    "posterior instrumentation 22842-22844 must not collide with anterior 22845-22847"
)
assert not (EXCLUSION_CORPECTOMY_CPTS & EXCLUSION_POSTERIOR_CERVICAL_CPTS)
assert not (EXCLUSION_CORPECTOMY_CPTS & EXCLUSION_ARTHROPLASTY_CPTS)
assert not (EXCLUSION_POSTERIOR_CERVICAL_CPTS & EXCLUSION_ARTHROPLASTY_CPTS)
assert TRACHEOSTOMY_CPT not in HEMATOMA_REOP_CPTS
assert not (
    HEMATOMA_REOP_CPTS
    & (PRINCIPAL_CPTS | EXCLUSION_CPTS | ANTERIOR_INSTRUMENTATION_CPTS)
), "a reoperation CPT is also an index-procedure CPT"
assert set(EXCLUSION_REASON) == set(EXCLUSION_CPTS), "EXCLUSION_REASON must cover EXCLUSION_CPTS"
assert set(EXCLUSION_REASON.values()) == set(EXCLUSION_REASON_ORDER)

# Seroma guards must be disjoint from what they guard, and must actually be reachable by a
# prefix. A seroma code that no prefix matches would mean the guard is decorative.
# Checked for EVERY variant, so a sensitivity definition cannot carry a broken guard.
assert not (SEROMA_ICD9_EXCLUDED & HEMATOMA_ICD9), "998.13 must not be a hematoma ICD-9 code"
for _v, _spec in HEMATOMA_ICD10_VARIANTS.items():
    assert not (_spec["seroma"] & _spec["exact"]), f"{_v}: a seroma code is in the exact set"
    for _c in _spec["seroma"]:
        assert _c.startswith(_spec["prefixes"]), (
            f"{_v}: seroma guard {_c!r} is unreachable: no hematoma prefix matches it, so "
            f"the guard is decorative and the prefix list has drifted"
        )
    # No prefix may be a prefix of another. This is the property that makes per-code
    # tallies a partition rather than a double count.
    for _a in _spec["prefixes"]:
        for _b in _spec["prefixes"]:
            assert _a == _b or not _b.startswith(_a), (
                f"{_v}: prefix {_a!r} subsumes prefix {_b!r}"
            )
    # The exact set and the prefix set must stay disjoint mechanisms.
    for _c in _spec["exact"]:
        assert not _c.startswith(_spec["prefixes"]), (
            f"{_v}: exact hematoma code {_c!r} is already covered by a prefix"
        )
for _a in DYSPHAGIA_ICD10_PREFIXES:
    for _b in DYSPHAGIA_ICD10_PREFIXES:
        assert _a == _b or not _b.startswith(_a), f"prefix {_a!r} subsumes prefix {_b!r}"

# Dysphagia and hematoma must never overlap: they are separate outcomes. Every variant.
for _c in DYSPHAGIA_ICD10_PLAN_CODES | set(DYSPHAGIA_ICD10_SUBTYPE):
    for _spec in HEMATOMA_ICD10_VARIANTS.values():
        assert not _c.startswith(_spec["prefixes"])
        assert _c not in _spec["exact"]
    assert _c.startswith(DYSPHAGIA_ICD10_PREFIXES), f"{_c!r} is outside the dysphagia family"
assert DYSPHAGIA_ICD10_PLAN_CODES <= set(DYSPHAGIA_ICD10_SUBTYPE)

# Rendered-text hygiene (CLAUDE.md hard rules 1 and 2) on everything this module exports.
# _EM_DASH is written as a \u2014 escape ON PURPOSE, matching the convention stated in
# `variable_dict.py`: no file in this project may contain a literal U+2014, so a repo-wide
# grep for the character stays clean and the ban is enforceable by search.
_EM_DASH = "\u2014"
for _d in (CODE_LABEL, EXCLUSION_REASON, DYSPHAGIA_ICD10_SUBTYPE, READMISSION_CODE_LABEL):
    for _k, _txt in _d.items():
        assert _EM_DASH not in _txt, f"label for {_k!r} contains an EM-dash"
        assert "_" not in _txt, f"label for {_k!r} leaks snake_case: {_txt!r}"

# Both ladders are pinned at import. Gate B (primary) first, then Gate A (reconciliation).
assert COHORT_LADDER_N == {
    "identified": 124344, "elective": 121210, "exclusions": 113469, "analytic": 112642,
}, "Gate B (primary, corrected) ladder drifted; see CLAUDE.md section 7.2"
assert COHORT_LADDER_N_PLAN == {
    "identified": 124266, "elective": 121142, "exclusions": 113411, "analytic": 112585,
}, "Gate A (plan replication) ladder drifted; see CLAUDE.md section 7.3"
assert COHORT_LADDER_EVENTS == {
    "identified": 642, "elective": 613, "exclusions": 548, "analytic": 541,
}, ("the shared event ladder drifted. It was re-pinned 2026-07-31 to the V2 primary "
    "outcome definition and every value was MEASURED (CLAUDE.md 7.4)")
assert COHORT_LADDER_EVENTS_BY_DEFINITION["V0"]["analytic"] == 537
assert COHORT_LADDER_EVENTS_BY_DEFINITION["V1"]["analytic"] == 534
assert COHORT_LADDER_EVENTS_BY_DEFINITION["V2"]["analytic"] == 541
for _v in HEMATOMA_DEFINITION_ORDER:
    assert set(COHORT_LADDER_EVENTS_BY_DEFINITION[_v]) == set(COHORT_LADDER_ORDER)
assert (
    set(COHORT_LADDER_N)
    == set(COHORT_LADDER_N_PLAN)
    == set(COHORT_LADDER_EVENTS)
    == set(COHORT_LADDER_ORDER)
)
# The delta is the 78 nonagenarians, minus those the later steps remove for other reasons.
assert COHORT_LADDER_NONAGENARIAN_DELTA["identified"] == 78, (
    "the Gate B minus Gate A delta at the identified step must be exactly the 78 patients "
    "whose AGE is the literal token '90+'"
)
for _step in COHORT_LADDER_ORDER:
    assert COHORT_LADDER_N[_step] > COHORT_LADDER_N_PLAN[_step], (
        f"Gate B must be strictly larger than Gate A at step {_step!r}"
    )
    assert 0 < COHORT_LADDER_NONAGENARIAN_DELTA[_step] <= 78

del _a, _b, _c, _d, _k, _txt, _step

# ---------------------------------------------------------------------------
# 11. Predicates. Every one is total: None / NaN / empty returns False, never raises.
# ---------------------------------------------------------------------------


def is_principal_cpt(code: Any) -> bool:
    """True if `code` is one of the three ACDF cohort-entry CPT codes."""
    return normalize_cpt(code) in PRINCIPAL_CPTS


def is_exclusion_cpt(code: Any) -> bool:
    """True if `code` disqualifies a case (corpectomy, posterior cervical, arthroplasty)."""
    return normalize_cpt(code) in EXCLUSION_CPTS


def exclusion_reason(code: Any) -> str:
    """Figure 1 exclusion-reason label for `code`, or '' if the code is not an exclusion."""
    return EXCLUSION_REASON.get(normalize_cpt(code), "")


def is_additional_level_cpt(code: Any) -> bool:
    """True if `code` is CPT 22552, the each-additional-interspace add-on."""
    return normalize_cpt(code) == ADDITIONAL_LEVEL_CPT


def is_anterior_instrumentation_cpt(code: Any) -> bool:
    """True if `code` is anterior instrumentation (22845 / 22846 / 22847)."""
    return normalize_cpt(code) in ANTERIOR_INSTRUMENTATION_CPTS


def is_tracheostomy_cpt(code: Any) -> bool:
    """True if `code` is CPT 31600, the reoperative-tracheostomy airway marker."""
    return normalize_cpt(code) == TRACHEOSTOMY_CPT


def is_hematoma_cpt(code: Any) -> bool:
    """True if `code` is a hematoma-evacuation reoperation CPT.

    Arm 1 of the three-system union, read from REOPORCPT1. Available all 13 years, which
    is why the year-by-year hematoma rate shows no gap at the ICD-9 to ICD-10 transition.
    """
    return normalize_cpt(code) in HEMATOMA_REOP_CPTS


def is_hematoma_icd9(code: Any) -> bool:
    """True if `code` is ICD-9 998.11 or 998.12. Seroma code 998.13 returns False.

    Arm 2 of the union, read from REOPORICD91 (2012-2020). Exact match: the ambiguous
    4-digit truncation 998.1 is not counted.
    """
    normalized = normalize_icd(code)
    if not normalized or normalized in SEROMA_ICD9_EXCLUDED:
        return False
    return normalized in HEMATOMA_ICD9


def is_hematoma_icd10_variant(code: Any, definition: str) -> bool:
    """`is_hematoma_icd10` under a named code-set variant. See section 5a.

    `definition` is 'V0', 'V1' or 'V2'. This is the function the Supplement S2 sensitivity
    rows use, so a sensitivity analysis never has to write a code literal of its own.
    """
    try:
        spec = HEMATOMA_ICD10_VARIANTS[definition]
    except KeyError:
        raise ValueError(
            f"unknown hematoma definition {definition!r}; expected one of "
            f"{list(HEMATOMA_DEFINITION_ORDER)}"
        ) from None
    normalized = normalize_icd(code)
    if not normalized or normalized in spec["seroma"]:
        return False
    if normalized in spec["exact"]:
        return True
    return normalized.startswith(spec["prefixes"])


def is_hematoma_icd10(code: Any) -> bool:
    """True if `code` is a hematoma or hemorrhage ICD-10 code, seroma members removed.

    Arm 3 of the union, read from REOPOR1ICD101 (2014-2024). The rule is

        (exact match OR prefix match) AND NOT in the seroma exclusion set

    with the seroma test applied FIRST. Prefix matching is required because the PUF stores
    truncated forms (M96.83, M96.84, L76.3, G97.6, G97.5 all occur), and the seroma guard
    is required because the same prefixes would otherwise sweep in M96.842, M96.843,
    L76.33, L76.34 and, from V1 onward, G97.63 and G97.64.

    This reads the PRIMARY definition, `HEMATOMA_DEFINITION` (section 5a). Use
    `is_hematoma_icd10_variant` for a sensitivity code set.
    """
    return is_hematoma_icd10_variant(code, HEMATOMA_DEFINITION)


def is_seroma_only_icd10(code: Any) -> bool:
    """True if `code` is one of the seroma codes deliberately kept out of the outcome."""
    return normalize_icd(code) in SEROMA_ICD10_EXCLUDED


def is_seroma_only_icd9(code: Any) -> bool:
    """True if `code` is ICD-9 998.13, seroma complicating a procedure."""
    return normalize_icd(code) in SEROMA_ICD9_EXCLUDED


def is_hematoma_any(cpt: Any = None, icd9: Any = None, icd10: Any = None) -> bool:
    """The primary outcome, as the union of all three arms for one reoperation slot.

    Call with the reoperation-1 triple:
        is_hematoma_any(row['REOPORCPT1'], row['REOPORICD91'], row['REOPOR1ICD101'])
    Any argument may be None or NaN; a case with no reoperation returns False.
    """
    return is_hematoma_cpt(cpt) or is_hematoma_icd9(icd9) or is_hematoma_icd10(icd10)


def is_hematoma_any_variant(cpt: Any = None, icd9: Any = None, icd10: Any = None,
                            definition: str = HEMATOMA_DEFINITION) -> bool:
    """`is_hematoma_any` under a named ICD-10 code-set variant. See section 5a.

    Only the ICD-10 arm differs between variants; the reoperation-CPT and ICD-9 arms are
    the same in all three.
    """
    return (is_hematoma_cpt(cpt) or is_hematoma_icd9(icd9)
            or is_hematoma_icd10_variant(icd10, definition))


def is_dysphagia_icd10(code: Any) -> bool:
    """True if `code` belongs to the R13.1 dysphagia family.

    Semantics, stated explicitly because R13.1 is a prefix of its own children: this is a
    SINGLE prefix test over the whole R13.1x family, returning one boolean per code. The
    family total is the number of records for which this returns True. It is NOT the sum
    of per-subcode counts, which would count every R13.10 record twice (once under R13.1
    and once under R13.10).

    R13 (unspecified) and R13.0 (aphagia) are outside the family and return False.
    Use `dysphagia_subtype` for the Table 5 breakdown; its buckets partition the family.
    """
    normalized = normalize_icd(code)
    if not normalized:
        return False
    return normalized.startswith(DYSPHAGIA_ICD10_PREFIXES)


def dysphagia_subtype(code: Any) -> str:
    """Table 5 subtype label for a dysphagia code, or '' if `code` is outside the family.

    Each family member maps to exactly one bucket, so tallying this column partitions the
    family total produced by `is_dysphagia_icd10`.
    """
    normalized = normalize_icd(code)
    if not is_dysphagia_icd10(normalized):
        return ""
    return DYSPHAGIA_ICD10_SUBTYPE.get(normalized, "Other dysphagia")


def count_levels(othercpt_values: Any) -> int:
    """Fusion levels = 1 + the number of CPT 22552 occurrences in `othercpt_values`.

    `othercpt_values` is the row's OTHERCPT1..OTHERCPT10 values, in any iterable form:
    a list, a tuple, a pandas Series, a numpy array, or the row slice `row[OTHERCPT_COLUMNS]`.
    None-padded and NaN-padded slots are normal and are simply not counted. A bare string
    is accepted and treated as a one-element sequence. None returns 1.

    The principal CPT is NOT part of the input and is NOT counted; the base level of 1 is
    the principal procedure itself. The result is uncapped, so 4 occurrences returns 5;
    use `level_stratum` for the 1 / 2 / >=3 analysis variable.

        count_levels([])                            -> 1
        count_levels(['22552'])                     -> 2
        count_levels(['22552', '22552'])            -> 3
        count_levels(['22552', None, '22845', nan]) -> 2
        count_levels(['22552.0'])                   -> 2
        count_levels(None)                          -> 1
    """
    if othercpt_values is None:
        return 1
    if isinstance(othercpt_values, (str, bytes)):
        othercpt_values = [othercpt_values]
    try:
        iterator: Iterable[Any] = iter(othercpt_values)
    except TypeError:
        iterator = [othercpt_values]
    extra = 0
    for value in iterator:
        if normalize_cpt(value) == ADDITIONAL_LEVEL_CPT:
            extra += 1
    return 1 + extra


def level_stratum(levels: Any) -> str:
    """Collapse a level count to the analysis stratum '1', '2' or '>=3'.

    Takes what `count_levels` returned. Anything that is not a positive integer returns
    LEVEL_STRATUM_UNKNOWN ('') rather than raising, so a malformed row is visible as
    missing instead of silently landing in a real stratum.
    """
    try:
        n = int(levels)
    except (TypeError, ValueError):
        return LEVEL_STRATUM_UNKNOWN
    if n < 1:
        return LEVEL_STRATUM_UNKNOWN
    if n == 1:
        return LEVEL_STRATUM_ORDER[0]
    if n == 2:
        return LEVEL_STRATUM_ORDER[1]
    return LEVEL_STRATUM_ORDER[2]


def any_code_matches(values: Any, predicate) -> bool:
    """True if `predicate` is True for at least one value in `values`. None-safe.

    Convenience for scanning a row's OTHERCPT1..10 slice for an exclusion or for anterior
    instrumentation without every caller rewriting the same None-guarded loop.
    """
    if values is None:
        return False
    if isinstance(values, (str, bytes)):
        values = [values]
    try:
        iterator: Iterable[Any] = iter(values)
    except TypeError:
        iterator = [values]
    return any(predicate(v) for v in iterator)


def code_label(code: Any) -> str:
    """Supplement S1 descriptor for a CPT or ICD code, or '' if none is recorded.

    Tries the CPT normalization first, then the ICD normalization, so one call serves both
    code systems. Codes in UNVERIFIED_DESCRIPTORS deliberately return ''.
    """
    as_cpt = normalize_cpt(code)
    if as_cpt in CODE_LABEL:
        return CODE_LABEL[as_cpt]
    as_icd = normalize_icd(code)
    if as_icd in CODE_LABEL:
        return CODE_LABEL[as_icd]
    return READMISSION_CODE_LABEL.get(as_icd, "")


# ---------------------------------------------------------------------------
# 12. Self-test: `python3 cpt_acdf.py`. Exits 0 on success, non-zero on any failure.
# ---------------------------------------------------------------------------

_NAN = float("nan")


def _self_test() -> None:
    checks = 0

    # --- 1. THE seroma cases, the reason prefix matching needs a guard -------------
    for code in ("M96.842", "M96.843", "L76.33", "L76.34"):
        assert is_hematoma_icd10(code) is False, f"seroma code {code} must NOT be a hematoma"
        assert is_seroma_only_icd10(code) is True, f"{code} must be flagged seroma-only"
        checks += 2
    for code in ("M96.840", "M96.841", "L76.31", "L76.32"):
        assert is_hematoma_icd10(code) is True, f"hematoma code {code} MUST match"
        assert is_seroma_only_icd10(code) is False
        checks += 2
    # The exact pair the specification calls out.
    assert is_hematoma_icd10("M96.842") is False and is_hematoma_icd10("M96.840") is True
    assert is_hematoma_icd10("L76.33") is False and is_hematoma_icd10("L76.31") is True
    checks += 4
    # ICD-9 seroma.
    assert is_hematoma_icd9("998.13") is False and is_seroma_only_icd9("998.13") is True
    assert is_hematoma_icd9("998.11") is True and is_hematoma_icd9("998.12") is True
    assert is_hematoma_icd9("998.1") is False, "the ambiguous 998.1 truncation is not counted"
    assert is_hematoma_icd9("998.59") is False
    checks += 6

    # --- 2. normalization: punctuation, case, whitespace, float-like CPT ----------
    for variant in ("M96.83", "m9683 ", "M9683", " m96.83", "\tM96.83\n", "m 96.83"):
        assert normalize_icd(variant) == "M9683", f"normalize_icd({variant!r}) failed"
        assert is_hematoma_icd10(variant) is True, f"{variant!r} must match after normalizing"
        checks += 2
    assert normalize_icd("998.12") == "99812" and normalize_icd("99812") == "99812"
    assert normalize_icd("t792xxa") == "T792XXA"
    assert is_hematoma_icd10("t79.2xxa") is True
    assert is_hematoma_icd10("T79.2XXA") is True
    checks += 4
    for variant in ("22551", " 22551 ", "22551.0", "22551.00", 22551, 22551.0):
        assert normalize_cpt(variant) == "22551", f"normalize_cpt({variant!r}) failed"
        assert is_principal_cpt(variant) is True
        checks += 2
    assert normalize_cpt("0095T") == "0095T" and normalize_cpt("0095t") == "0095T"
    assert normalize_cpt("22551.5") == "22551.5", "a non-zero fraction is not a CPT; leave it"
    checks += 3

    # --- 3. prefix families, and what must NOT be swept in ------------------------
    # PRIMARY definition (V2). G97.63 / G97.64 are seroma and are guarded; T81.0 does not
    # exist in ICD-10-CM and is gone; G97.51 / G97.52 are the hemorrhage arm and are IN.
    for code in ("M96.83", "M96.830", "M96.831", "M96.84", "L76.2", "L76.21", "L76.22",
                 "L76.3", "G97.6", "G97.61", "G97.62", "G97.5", "G97.51", "G97.52"):
        assert is_hematoma_icd10(code) is True, f"{code} should match a hematoma prefix"
        checks += 1
    for code in ("T81.31XA", "T81.32XA", "T81.4XXA", "L76.01", "L76.02", "L76.11",
                 "M96.811", "M96.821", "M96.820", "M96.89", "K91.89", "T79.A0XA",
                 "R13.10", "T81.0", "T81.0XXA", "G97.63", "G97.64"):
        assert is_hematoma_icd10(code) is False, f"{code} must NOT be counted as hematoma"
        checks += 1
    # exact-only codes. M96.820 left the exact set at V1; M96.810 stays.
    assert is_hematoma_icd10("M96.810") is True and is_hematoma_icd10("M96.820") is False
    assert is_hematoma_icd10("M96.81") is False and is_hematoma_icd10("M968") is False
    checks += 4

    # --- 3a. the three code-set variants behave as section 5a states --------------
    assert HEMATOMA_DEFINITION == "V2"
    assert is_hematoma_icd10_variant("T81.0", "V0") is True
    assert is_hematoma_icd10_variant("T81.0", "V1") is False
    assert is_hematoma_icd10_variant("T81.0", "V2") is False
    assert is_hematoma_icd10_variant("M96.820", "V0") is True
    assert is_hematoma_icd10_variant("M96.820", "V1") is False
    assert is_hematoma_icd10_variant("M96.820", "V2") is False
    checks += 6
    for code in ("G97.63", "G97.64"):
        assert is_hematoma_icd10_variant(code, "V0") is True, f"{code} is unguarded in V0"
        assert is_hematoma_icd10_variant(code, "V1") is False
        assert is_hematoma_icd10_variant(code, "V2") is False
        checks += 3
    for code in ("G97.5", "G97.51", "G97.52"):
        assert is_hematoma_icd10_variant(code, "V0") is False
        assert is_hematoma_icd10_variant(code, "V1") is False
        assert is_hematoma_icd10_variant(code, "V2") is True, f"{code} is the V2 addition"
        checks += 3
    # Codes common to all three variants must not move.
    for code in ("M96.830", "L76.32", "G97.61", "T79.2XXA", "M96.810"):
        assert (is_hematoma_icd10_variant(code, "V0")
                is is_hematoma_icd10_variant(code, "V1")
                is is_hematoma_icd10_variant(code, "V2") is True), f"{code} moved"
        checks += 1
    for code in ("M96.842", "L76.34"):
        assert (is_hematoma_icd10_variant(code, "V0")
                is is_hematoma_icd10_variant(code, "V1")
                is is_hematoma_icd10_variant(code, "V2") is False), f"{code} moved"
        checks += 1
    # The variant API is total and rejects an unknown name loudly.
    assert is_hematoma_icd10_variant(None, "V2") is False
    assert is_hematoma_icd10_variant(_NAN, "V0") is False
    try:
        is_hematoma_icd10_variant("M96.830", "V9")
    except ValueError:
        checks += 1
    else:                                            # pragma: no cover
        raise AssertionError("an unknown definition name must raise")
    assert is_hematoma_any_variant(None, None, "G97.51", definition="V2") is True
    assert is_hematoma_any_variant(None, None, "G97.51", definition="V0") is False
    assert is_hematoma_any_variant("10140", None, "G97.51", definition="V0") is True
    assert is_hematoma_any_variant(None, None, "G97.51") is True, "default is the primary"
    checks += 6
    # The pinned per-variant event ladders, and the direction of each correction.
    assert (COHORT_LADDER_EVENTS_BY_DEFINITION["V1"]["analytic"]
            < COHORT_LADDER_EVENTS_BY_DEFINITION["V0"]["analytic"]
            < COHORT_LADDER_EVENTS_BY_DEFINITION["V2"]["analytic"])
    assert COHORT_LADDER_EVENTS == COHORT_LADDER_EVENTS_BY_DEFINITION["V2"]
    checks += 2
    # CPT arm
    for code in ("10140", "21501", "35800", "20005", " 10140 ", "10140.0"):
        assert is_hematoma_cpt(code) is True
        checks += 1
    for code in ("49000", "31600", "22551", "11042"):
        assert is_hematoma_cpt(code) is False
        checks += 1
    # the union
    assert is_hematoma_any("10140", None, None) is True
    assert is_hematoma_any(None, "998.12", None) is True
    assert is_hematoma_any(None, None, "l76.32") is True
    assert is_hematoma_any(None, None, "L76.34") is False, "seroma must not fire the union"
    assert is_hematoma_any(None, None, None) is False
    assert is_hematoma_any() is False
    checks += 6

    # --- 4. dysphagia semantics: family membership, and no double counting --------
    for code in ("R13.1", "R13.10", "R13.11", "R13.12", "R13.13", "R13.14", "R13.19",
                 "r1310", " R13.10 "):
        assert is_dysphagia_icd10(code) is True, f"{code} is in the R13.1 family"
        checks += 1
    for code in ("R13", "R13.0", "R130", "K21.9", "M96.840", ""):
        assert is_dysphagia_icd10(code) is False, f"{code} is outside the R13.1 family"
        checks += 1
    # A single record yields exactly one prefix hit, so the family total cannot be a
    # double count of its own sub-codes.
    for code in ("R13.1", "R13.10", "R13.12", "R13.19"):
        hits = [p for p in DYSPHAGIA_ICD10_PREFIXES if normalize_icd(code).startswith(p)]
        assert len(hits) == 1, f"{code} matched {len(hits)} prefixes; the family would double count"
        checks += 1
    # Subtype buckets partition the family: one bucket per code, none empty, none shared.
    seen = {}
    for code in ("R13.1", "R13.10", "R13.11", "R13.12", "R13.13", "R13.14", "R13.19"):
        label = dysphagia_subtype(code)
        assert label, f"{code} has no subtype bucket"
        assert label not in seen, f"{code} shares a bucket with {seen.get(label)}"
        seen[label] = code
        checks += 1
    assert dysphagia_subtype("R13.0") == "" and dysphagia_subtype(None) == ""
    assert dysphagia_subtype("M96.840") == ""
    checks += 3

    # --- 5. level counting -------------------------------------------------------
    assert count_levels([]) == 1, "no 22552 means a single-level fusion"
    assert count_levels(["22552"]) == 2
    assert count_levels(["22552", "22552"]) == 3, "2 extra 22552 means 3 levels"
    assert count_levels(["22552", "22552", "22552"]) == 4
    checks += 4
    # None-padded and NaN-padded OTHERCPT lists, full width
    padded = ["22552", "22845", None, _NAN, "", "NULL", "-99", None, None, None]
    assert count_levels(padded) == 2, f"padded list gave {count_levels(padded)}"
    padded2 = ["22552", "22552", None, None, None, None, None, None, None, None]
    assert count_levels(padded2) == 3
    assert count_levels([None] * 10) == 1
    assert count_levels([_NAN] * 10) == 1
    checks += 4
    # float-like CPT strings in the OTHERCPT slots
    assert count_levels(["22552.0", "22552.00"]) == 3
    assert count_levels([22552.0, 22552]) == 3
    checks += 2
    # degenerate inputs
    assert count_levels(None) == 1
    assert count_levels("22552") == 2, "a bare string is treated as one slot"
    assert count_levels("22551") == 1
    assert count_levels(tuple()) == 1
    assert count_levels(iter(["22552"])) == 2
    checks += 5
    # strata
    assert level_stratum(1) == "1" and level_stratum(2) == "2"
    assert level_stratum(3) == ">=3" and level_stratum(4) == ">=3" and level_stratum(9) == ">=3"
    assert level_stratum(count_levels(["22552", "22552"])) == ">=3"
    assert level_stratum(0) == LEVEL_STRATUM_UNKNOWN
    assert level_stratum(None) == LEVEL_STRATUM_UNKNOWN
    assert level_stratum("banana") == LEVEL_STRATUM_UNKNOWN
    assert level_stratum(_NAN) == LEVEL_STRATUM_UNKNOWN
    assert set(LEVEL_STRATUM_ORDER) == {"1", "2", ">=3"}
    checks += 9

    # --- 6. every predicate is total on None / NaN / empty / junk ----------------
    predicates = (
        is_principal_cpt, is_exclusion_cpt, is_additional_level_cpt,
        is_anterior_instrumentation_cpt, is_tracheostomy_cpt,
        is_hematoma_cpt, is_hematoma_icd9, is_hematoma_icd10,
        is_seroma_only_icd9, is_seroma_only_icd10, is_dysphagia_icd10,
    )
    for predicate in predicates:
        for bad in (None, _NAN, "", "   ", "NULL", "nan", "NaN", "-99", "N/A", ".", "<NA>", 0):
            result = predicate(bad)
            assert result is False, f"{predicate.__name__}({bad!r}) returned {result!r}, want False"
            checks += 1
    assert exclusion_reason(None) == "" and exclusion_reason(_NAN) == ""
    assert code_label(None) == "" and code_label(_NAN) == ""
    assert any_code_matches(None, is_exclusion_cpt) is False
    assert any_code_matches([None, _NAN, ""], is_exclusion_cpt) is False
    assert any_code_matches([None, "22856"], is_exclusion_cpt) is True
    assert any_code_matches("22856", is_exclusion_cpt) is True
    checks += 6

    # --- 7. exclusions, instrumentation, tracheostomy ----------------------------
    for code in ("63081", "63082", "22548", "22600", "22842", "22843", "22844",
                 "63001", "63015", "63045", "63047", "22856"):
        assert is_exclusion_cpt(code) is True, f"{code} must be an exclusion"
        assert exclusion_reason(code) in EXCLUSION_REASON_ORDER
        checks += 2
    for code in ("22551", "22554", "63075", "22552", "22845", "63048", "20930"):
        assert is_exclusion_cpt(code) is False, f"{code} must NOT be an exclusion"
        checks += 1
    assert exclusion_reason("63081") == "Concurrent corpectomy"
    # CLAUDE.md 6d.6 / 6j.2: the banned wording must not come back by accident.
    assert exclusion_reason("22600") == (
        "Concurrent posterior spinal decompression or instrumentation")
    assert not any("posterior cervical" in v.lower()
                   for v in EXCLUSION_REASON.values()), (
        "CLAUDE.md 6d.6 bans 'posterior cervical' as a rendered exclusion label")
    assert exclusion_reason("22856") == "Cervical disc arthroplasty"
    assert exclusion_reason("22551") == ""
    checks += 4
    for code in ("22845", "22846", "22847", " 22845.0 "):
        assert is_anterior_instrumentation_cpt(code) is True
        checks += 1
    for code in ("22842", "22843", "22844", "22551"):
        assert is_anterior_instrumentation_cpt(code) is False, f"{code} is posterior, not anterior"
        checks += 1
    assert is_tracheostomy_cpt("31600") is True and is_tracheostomy_cpt("31601") is False
    assert is_tracheostomy_cpt(" 31600.0 ") is True
    checks += 3
    for code in ("22551", "22554", "63075"):
        assert is_principal_cpt(code) is True
        checks += 1
    for code in ("22552", "22856", "63081", "27130"):
        assert is_principal_cpt(code) is False
        checks += 1
    assert is_additional_level_cpt("22552") is True and is_additional_level_cpt("22551") is False
    checks += 2

    # --- 8. no code lives in two mutually exclusive sets --------------------------
    exclusive_groups = {
        "principal": set(PRINCIPAL_CPTS),
        "exclusion": set(EXCLUSION_CPTS),
        "additional level": {ADDITIONAL_LEVEL_CPT},
        "anterior instrumentation": set(ANTERIOR_INSTRUMENTATION_CPTS),
        "tracheostomy": {TRACHEOSTOMY_CPT},
        "hematoma reoperation": set(HEMATOMA_REOP_CPTS),
    }
    names = sorted(exclusive_groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = exclusive_groups[a] & exclusive_groups[b]
            assert not overlap, f"CPT sets '{a}' and '{b}' both contain {sorted(overlap)}"
            checks += 1
    icd_groups = {
        "hematoma ICD-9": set(HEMATOMA_ICD9),
        "seroma ICD-9": set(SEROMA_ICD9_EXCLUDED),
        "hematoma ICD-10 exact": set(HEMATOMA_ICD10_EXACT),
        "seroma ICD-10": set(SEROMA_ICD10_EXCLUDED),
        "dysphagia ICD-10": set(DYSPHAGIA_ICD10_SUBTYPE),
    }
    names = sorted(icd_groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = icd_groups[a] & icd_groups[b]
            assert not overlap, f"ICD sets '{a}' and '{b}' both contain {sorted(overlap)}"
            checks += 1
    # No hematoma code may also be a dysphagia code, under either mechanism.
    for code in DYSPHAGIA_ICD10_SUBTYPE:
        assert is_hematoma_icd10(code) is False
        checks += 1
    for code in HEMATOMA_ICD10_EXACT | set(HEMATOMA_ICD10_PREFIXES) | SEROMA_ICD10_EXCLUDED:
        assert is_dysphagia_icd10(code) is False
        checks += 1

    # --- 9. descriptors -----------------------------------------------------------
    documented = (
        PRINCIPAL_CPTS | EXCLUSION_CPTS | HEMATOMA_REOP_CPTS | ANTERIOR_INSTRUMENTATION_CPTS
        | {ADDITIONAL_LEVEL_CPT, TRACHEOSTOMY_CPT}
        | HEMATOMA_ICD9 | SEROMA_ICD9_EXCLUDED
        | HEMATOMA_ICD10_EXACT | SEROMA_ICD10_EXCLUDED
        | frozenset(HEMATOMA_ICD10_PREFIXES) | set(DYSPHAGIA_ICD10_SUBTYPE)
    )
    for code in documented:
        assert code in CODE_LABEL or code in UNVERIFIED_DESCRIPTORS, (
            f"{code} has neither a descriptor nor an entry in UNVERIFIED_DESCRIPTORS"
        )
        checks += 1
    assert code_label("22551").startswith("Anterior cervical")
    assert code_label("998.12") == "Hematoma complicating a procedure"
    # The five descriptors corrected 2026-07-31 against the CMS FY2026 order file. Each was
    # previously published with "Descriptor verified: Yes" and each was wrong.
    assert code_label("m96.84") == (
        "Postprocedural hematoma and seroma of a musculoskeletal structure following "
        "a procedure")
    assert code_label("L76.3") == (
        "Postprocedural hematoma and seroma of skin and subcutaneous tissue following "
        "a procedure")
    assert code_label("G97.6") == (
        "Postprocedural hematoma and seroma of a nervous system organ or structure "
        "following a procedure")
    assert "hemorrhage" not in code_label("G97.6").lower(), (
        "G97.6 is hematoma and seroma; hemorrhage of a nervous system organ is G97.5-"
    )
    assert code_label("T79.2XXA") == (
        "Traumatic secondary and recurrent hemorrhage and seroma, initial encounter")
    assert code_label("M96.820") == (
        "Accidental puncture and laceration of a musculoskeletal structure during "
        "a musculoskeletal system procedure"), "M96.820 is verified as of 2026-07-31"
    assert not UNVERIFIED_DESCRIPTORS, "nothing is awaiting a tabular-list check"
    assert normalize_icd("T81.0") in NONEXISTENT_ICD10
    assert not (NONEXISTENT_ICD10 & UNVERIFIED_DESCRIPTORS), (
        "a code cannot be both verified-not-to-exist and awaiting verification"
    )
    assert not (NONEXISTENT_ICD10 & frozenset(CODE_LABEL)), (
        "a code that does not exist must not carry a descriptor"
    )
    for _v in HEMATOMA_DEFINITION_ORDER[1:]:
        assert not (NONEXISTENT_ICD10 & frozenset(HEMATOMA_ICD10_VARIANTS[_v]["prefixes"])), (
            f"{_v} still carries a code that does not exist in ICD-10-CM"
        )
    checks += 5
    assert code_label("T81.0") == "", "T81.0 is not an ICD-10-CM code and has no descriptor"
    assert code_label("99999") == ""
    checks += 11

    # --- 10. BOTH verified ladders are internally consistent -----------------------
    # Gate B (COHORT_LADDER_N) is primary; Gate A (COHORT_LADDER_N_PLAN) is the
    # --replicate-plan reconciliation target. Both are checked, and so is the shared
    # event ladder and the nonagenarian delta between them (CLAUDE.md section 7).
    for gate, ladder in (("B (primary, corrected)", COHORT_LADDER_N),
                         ("A (plan replication)", COHORT_LADDER_N_PLAN)):
        previous_n, previous_e = None, None
        for step in COHORT_LADDER_ORDER:
            n, e = ladder[step], COHORT_LADDER_EVENTS[step]
            assert n > 0 and e > 0 and e < n, f"gate {gate} step '{step}' is not a cohort"
            if previous_n is not None:
                assert n <= previous_n, f"gate {gate} step '{step}' grew the cohort"
                assert e <= previous_e, f"gate {gate} step '{step}' grew the event count"
            previous_n, previous_e = n, e
            checks += 3

    # The two ladders are distinct objects with the pinned values, and the delta between
    # them is the nonagenarian cohort. A future edit that collapses them fails here.
    assert COHORT_LADDER_N is not COHORT_LADDER_N_PLAN
    assert COHORT_LADDER_N != COHORT_LADDER_N_PLAN, (
        "COHORT_LADDER_N (Gate B) and COHORT_LADDER_N_PLAN (Gate A) must not be equal"
    )
    assert COHORT_LADDER_N["analytic"] == 112642, "Gate B analytic cohort must be 112,642"
    assert COHORT_LADDER_N_PLAN["analytic"] == 112585, "Gate A analytic cohort must be 112,585"
    assert COHORT_LADDER_EVENTS["analytic"] == 541, (
        "the primary (V2) analytic event count is 541. 537 is V0 and 534 is V1; both are "
        "published as Supplement S2 sensitivity rows"
    )
    checks += 5

    # ONE event dict, shared, because the event counts are identical on both ladders: the
    # 78 restored nonagenarians carry zero hematoma events. Re-measured 2026-07-30.
    assert COHORT_LADDER_NONAGENARIAN_DELTA == {
        "identified": 78, "elective": 68, "exclusions": 58, "analytic": 57,
    }, "the Gate B minus Gate A delta drifted"
    assert sum(COHORT_LADDER_NONAGENARIAN_DELTA.values()) > 0
    for step in COHORT_LADDER_ORDER:
        # the delta never carries an event, which is why one event dict serves both gates
        assert COHORT_LADDER_N[step] - COHORT_LADDER_N_PLAN[step] > 0
        checks += 1
    checks += 2

    print(f"cpt_acdf self-test: {checks} checks PASS")
    print(f"  principal CPTs ................. {sorted(PRINCIPAL_CPTS)}")
    print(f"  exclusion CPTs ................. {len(EXCLUSION_CPTS)} codes, "
          f"{len(EXCLUSION_REASON_ORDER)} reasons")
    print(f"  additional-level CPT ........... {ADDITIONAL_LEVEL_CPT} "
          f"(strata {list(LEVEL_STRATUM_ORDER)})")
    print(f"  anterior instrumentation ....... {sorted(ANTERIOR_INSTRUMENTATION_CPTS)}")
    print(f"  tracheostomy marker ............ {TRACHEOSTOMY_CPT}")
    print(f"  hematoma reoperation CPTs ...... {sorted(HEMATOMA_REOP_CPTS)}")
    print(f"  hematoma ICD-9 ................. {sorted(HEMATOMA_ICD9)}")
    print(f"  hematoma ICD-10 prefixes ....... {list(HEMATOMA_ICD10_PREFIXES)}")
    print(f"  hematoma ICD-10 exact .......... {sorted(HEMATOMA_ICD10_EXACT)}")
    print(f"  seroma guard (ICD-10 / ICD-9) .. {sorted(SEROMA_ICD10_EXCLUDED)} / "
          f"{sorted(SEROMA_ICD9_EXCLUDED)}")
    print(f"  outcome definition ............. {HEMATOMA_DEFINITION} (PRIMARY), "
          f"{HEMATOMA_ICD10_VARIANTS[HEMATOMA_DEFINITION]['label']}")
    for _v in HEMATOMA_DEFINITION_ORDER:
        print(f"    {_v} analytic events ......... "
              f"{COHORT_LADDER_EVENTS_BY_DEFINITION[_v]['analytic']}"
              + ("   <- PRIMARY" if _v == HEMATOMA_DEFINITION else "   (sensitivity)"))
    print(f"  dysphagia ICD-10 prefixes ...... {list(DYSPHAGIA_ICD10_PREFIXES)} "
          f"({len(DYSPHAGIA_ICD10_SUBTYPE)} subtype buckets)")
    print(f"  ladder Gate B (PRIMARY) ........ "
          + " -> ".join(f"{COHORT_LADDER_N[s]:,}/{COHORT_LADDER_EVENTS[s]}"
                        for s in COHORT_LADDER_ORDER))
    print(f"  ladder Gate A (replicate-plan) . "
          + " -> ".join(f"{COHORT_LADDER_N_PLAN[s]:,}/{COHORT_LADDER_EVENTS[s]}"
                        for s in COHORT_LADDER_ORDER))
    print(f"  nonagenarian delta (B minus A) . "
          + " -> ".join(f"{COHORT_LADDER_NONAGENARIAN_DELTA[s]}"
                        for s in COHORT_LADDER_ORDER)
          + "   (0 events at every step, so one shared event ladder)")


if __name__ == "__main__":
    _self_test()
