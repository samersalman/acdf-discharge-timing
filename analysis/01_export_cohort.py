#!/usr/bin/env python3
"""01_export_cohort.py -- pull the identified ACDF cohort out of the NSQIP warehouse.

This is the ACDF DISCHARGE-TIMING project (a spin-out of `projects/ACDF-hematoma/`). The
cohort is IDENTICAL to the parent's by design: the two manuscripts report the same ladder,
and a discrepancy between them would be visible to any reviewer holding both papers. So
this file is the parent's exporter, ported, with the export contract widened by exactly one
column.

One bundle comes out of this step:

    exports/acdf-discharge-timing/    principal CPT in `cpt_acdf.PRINCIPAL_CPTS`,
                                      puf_year 2012-2024.
                                      EXPECTED_N = 124,344 rows x 143 columns.

  (The three principal codes are deliberately NOT written out above. CLAUDE.md hard rule 5
  makes `cpt_acdf.py` the only place in this project where a CPT or ICD code may appear as
  a literal, and it names description strings explicitly, so the docstring names the
  constant instead.)

THE 143-COLUMN CONTRACT, AND WHY IT IS 143 AND NOT THE PARENT'S 142
  `vars/acdf_discharge_timing.txt` is the parent's `vars/acdf_hematoma.txt` with exactly
  one addition, `INOUT`: 135 -> 136 requested columns, plus the 7 columns export.py
  auto-injects, giving 136 + 7 = 143. Nothing else in the list moved, so the bundle is the
  parent's bundle plus one column and the cohort cannot fork.

WHY `INOUT` IS HERE AT ALL, GIVEN THE PARENT BANS IT
  `INOUT` is the NSQIP inpatient/outpatient BILLING label. It records how a case was booked,
  not how long the patient stayed, and it disagrees with true postoperative length of stay in
  the large majority of cases it labels 'Outpatient'. The parent project bans it outright
  (framing rule 3 of `projects/ACDF-hematoma/v1/CLAUDE.md`: true postoperative length of stay
  only, never the `INOUT` billing label) and hard-bans it in that project's preflight, so it
  cannot leak into that manuscript. That ban remains in force there and this file does not
  touch it.

  THIS project's second headline finding IS that disagreement (`PLAN.md` section 6.4): the
  2x2 of `INOUT` against true postoperative length of stay, its sensitivity / specificity /
  PPV / NPV / kappa, the year-by-year disagreement rate, and a published
  outpatient-versus-inpatient complication comparison re-estimated under both exposure
  definitions on the same cohort. None of that is computable without the column. So the ban
  is lifted HERE, narrowly, and the preflight below asserts the OPPOSITE of the parent's:
  `INOUT` must be PRESENT in the variable list, and its absence is a hard failure rather than
  a silent loss of an analysis.

Strategy (idempotent; `python3 01_export_cohort.py`):

  1. Import the principal CPT set from `cpt_acdf.PRINCIPAL_CPTS` -- the single source of
     truth (CLAUDE.md hard rule 5). No CPT code is retyped in this file.
  2. Invoke the warehouse exporter `nsqip-db/export.py` as a READ-ONLY SUBPROCESS, run
     with cwd = the warehouse directory so its relative paths resolve. It is never
     imported, and nothing this project owns writes into `nsqip-db/`. The exporter writes
     its own bundle to `nsqip-db/exports/<slug>/`, which is a generated, .gitignore'd
     artifact directory belonging to the exporter, not warehouse data. `--save-spec` is
     deliberately NOT passed, because that would write into the COMMITTED
     `nsqip-db/cohorts/` directory; the resolved spec still ships inside the bundle as
     `cohort_spec.yaml`.
  3. Copy the six-file bundle into this project's `exports/acdf-discharge-timing/`.
  4. Verify HARD: exact row count against `cpt_acdf.COHORT_LADDER_N['identified']`,
     exactly 143 columns (136 listed in vars/acdf_discharge_timing.txt + 7 auto-injected),
     manifest cross-checked against an independent count of cohort.csv, no `--where`
     predicate in the spec, no principal CPT outside the requested set, AGE >= 18 on every
     row, and `INOUT` present, populated and carrying no level beyond 'Inpatient' and
     'Outpatient'.

NO `--where` PREDICATE IS PASSED AT ALL.
  The plan's "identified" rung is `principal CPT in the set AND AGE >= 18`. Measured
  read-only against the curated `main` view on 2026-07-30: the CPT-and-year pull already
  returns exactly 124,344 rows, with `AGE` SMALLINT, 0 NULL, min 18, max 90. There is not
  one ACDF case under 18 in 13 years, so the age criterion is non-binding and needs no SQL.
  This keeps the export a pure, unfiltered identified pull (CLAUDE.md hard rule 6: any
  `--where` would have to be NULL-safe, and the safest NULL-safe predicate is none). The
  elective restriction, the procedure exclusions, the ASA / sepsis exclusions and every
  recode belong to `02_build_cohort.py`, so the STROBE flow keeps every rung.

WHY THE ROW-COUNT CHECK IS EXACT AND NOT A TOLERANCE
  124,344 is warehouse-verified ground truth pinned in `cpt_acdf.COHORT_LADDER_N`
  (CLAUDE.md 8.2, Gate B). A mismatch means the CPT filter, the year filter or the
  warehouse itself changed, and every downstream number is suspect -- so it raises
  SystemExit rather than printing a warning. CLAUDE.md 8.5: a mismatch is a stop
  condition, not a number to overwrite. This is also Gate 1 (CLAUDE.md 8.3): adding a
  column must not move a single row count, and this file is where that is first testable.

NO RECODING HAPPENS HERE. The warehouse emits human-readable strings and SQL NULLs.
Note in particular that `PRSEPIS` stores the literal string 'None' for "no preoperative
sepsis"; `02_build_cohort.py` must read this CSV with `keep_default_na=False` or the
sepsis filter inverts (CLAUDE.md 9.8).

Usage
-----
    python3 01_export_cohort.py            # reuse a valid bundle, else export
    python3 01_export_cohort.py --force    # always re-run the warehouse export
    python3 01_export_cohort.py --dry-run  # print the exporter's resolved SQL, write nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
from collections import Counter

# --- paths -----------------------------------------------------------------
PROJECT_DIR = pathlib.Path(__file__).resolve().parent              # this analysis/ directory
# The NSQIP DuckDB warehouse. Not distributed with this repository: it is built from
# the ACS-NSQIP Participant Use Data File, which requires a Data Use Agreement.
# Point NSQIP_WAREHOUSE at your own copy, or place it at ~/nsqip-db.
WAREHOUSE_DIR = pathlib.Path(
    os.environ.get("NSQIP_WAREHOUSE", pathlib.Path.home() / "nsqip-db")
).expanduser().resolve()
WAREHOUSE_EXPORT_PY = WAREHOUSE_DIR / "export.py"
PROJECT_EXPORTS_DIR = PROJECT_DIR / "exports"

sys.path.insert(0, str(PROJECT_DIR))
import cpt_acdf  # noqa: E402  (the CPT taxonomy; single source of truth)

# --- export contract (verified in .build_facts.md sections 3 and 4) --------
SLUG = "acdf-discharge-timing"
FAMILY = "main"
YEARS = "2012-2024"
VAR_FILE = PROJECT_DIR / "vars" / "acdf_discharge_timing.txt"

N_REQUESTED_COLS = 136                # vars/acdf_discharge_timing.txt (parent's 135 + INOUT)
AUTO_INJECTED_COLS = [                # export.py:38, config.IDENTITY_COLS + case_uid
    "CASEID", "PUFYEAR", "OPERYR", "CPT", "PRNCPTX", "SOURCE_FAMILY", "CASE_UID",
]
EXPECTED_COLS = N_REQUESTED_COLS + len(AUTO_INJECTED_COLS)          # 143

# Gate B, pinned in cpt_acdf. Never retyped here (CLAUDE.md 8.2).
EXPECTED_N = cpt_acdf.COHORT_LADDER_N["identified"]                  # 124,344

MIN_AGE = 18                          # the plan's inclusion criterion, asserted not applied

# --- the INOUT contract (CLAUDE.md 4.4, .build_facts.md sections 3 and 4) --
# The billing-label column this project's section 6.4 critique is built on. Measured
# read-only on the curated `main` view for puf_year 2012-2024 on 2026-07-31: VARCHAR,
# present in all 13 years, exactly two non-null levels plus one SQL NULL database-wide
# (Inpatient 6,457,523 / Outpatient 5,357,395 / NULL 1, the NULL in 2013).
INOUT_COL = "INOUT"
INOUT_LEVELS = ("Inpatient", "Outpatient")

BUNDLE_FILES = [
    "cohort.csv",
    "data_dictionary.csv",
    "query.sql",
    "cohort_spec.yaml",
    "manifest.json",
    "README.md",
]

# PRNCPTX procedure text is long; raise the cap so a wide free-text field can never
# truncate a verification count.
csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Var-file preflight
# ---------------------------------------------------------------------------
def _read_var_file(path: pathlib.Path):
    """The same parse the exporter uses: strip '#' comments, drop blanks, uppercase."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line.upper())
    return out


def _preflight() -> list:
    if not WAREHOUSE_EXPORT_PY.exists():
        raise SystemExit(f"[01] warehouse export.py not found at {WAREHOUSE_EXPORT_PY}")
    if not VAR_FILE.exists():
        raise SystemExit(f"[01] var file not found at {VAR_FILE}")

    codes = _read_var_file(VAR_FILE)
    if len(codes) != len(set(codes)):
        dupes = sorted({c for c in codes if codes.count(c) > 1})
        raise SystemExit(f"[01] {VAR_FILE.name} has duplicate entries: {dupes}")
    if len(codes) != N_REQUESTED_COLS:
        raise SystemExit(
            f"[01] {VAR_FILE.name} lists {len(codes)} columns, expected "
            f"{N_REQUESTED_COLS}. The {EXPECTED_COLS}-column contract depends on this."
        )
    leaked = sorted(set(codes) & set(AUTO_INJECTED_COLS))
    if leaked:
        raise SystemExit(
            f"[01] {VAR_FILE.name} lists auto-injected column(s) {leaked}. "
            f"export.py injects these; listing them is an ERROR (.build_facts.md 4)."
        )
    # NOTE THE MISSING ENTRY. The parent's ban list is {INOUT, FNSTATUS1, PRPT, HDISDT};
    # this one is {FNSTATUS1, PRPT, HDISDT}. The INOUT ban is LIFTED FOR THIS PROJECT ONLY,
    # because the `INOUT` versus true length-of-stay measurement critique (PLAN.md 6.4) is
    # one of this paper's two headline findings and is not computable without the column.
    # The ban REMAINS IN FORCE in `projects/ACDF-hematoma/`, whose framing rule 3 keeps the
    # billing label out of that manuscript entirely; do not carry this relaxation back.
    # The other three stay banned for the same reasons as in the parent: FNSTATUS1 is 100%
    # NULL database-wide in the only three years it exists, PRPT is 99% NULL in this cohort,
    # and HDISDT is a four-digit YEAR rather than a date.
    banned = sorted(set(codes) & {"FNSTATUS1", "PRPT", "HDISDT"})
    if banned:
        raise SystemExit(
            f"[01] {VAR_FILE.name} lists deliberately-omitted column(s) {banned}. "
            f"FNSTATUS1 and PRPT are dead columns; HDISDT is a year, not a date."
        )
    # The POSITIVE assertion, and the reason it exists: the parent bans this column, so the
    # single most likely bad edit to this file is someone re-syncing it against the parent's
    # list and quietly deleting INOUT. That would not fail any count check by itself if the
    # width constant were "fixed" alongside it, and the section 6.4 analysis would simply
    # vanish. Assert it is present, by name.
    if INOUT_COL not in codes:
        raise SystemExit(
            f"[01] {VAR_FILE.name} does NOT list {INOUT_COL}. This project REQUIRES it: "
            f"the {INOUT_COL} versus true postoperative length-of-stay measurement "
            f"critique (PLAN.md section 6.4) is one of this paper's two headline findings "
            f"and 05_inout.py cannot run without the column. It is banned in "
            f"projects/ACDF-hematoma/ and exported here on purpose. Re-add it to the "
            f"'Outcome: mortality, length of stay, disposition' block."
        )
    print(f"[01]   var-file OK: {VAR_FILE}")
    print(f"[01]     {len(codes)} requested + {len(AUTO_INJECTED_COLS)} auto-injected "
          f"= {EXPECTED_COLS} expected columns")
    print(f"[01]     {INOUT_COL} present in the variable list (required by PLAN.md 6.4; "
          f"banned in the parent project)")
    return codes


def _principal_cpts() -> list:
    """The --cpt arguments, built from the taxonomy. Never hard-coded (hard rule 5)."""
    codes = sorted(cpt_acdf.PRINCIPAL_CPTS)
    if len(codes) != 3:
        raise SystemExit(
            f"[01] cpt_acdf.PRINCIPAL_CPTS has {len(codes)} codes, expected 3"
        )
    for c in codes:
        if not cpt_acdf.is_principal_cpt(c):
            raise SystemExit(f"[01] cpt_acdf is inconsistent: {c!r} fails is_principal_cpt")
    return codes


# ---------------------------------------------------------------------------
# Warehouse exporter subprocess (READ-ONLY; never imported)
# ---------------------------------------------------------------------------
def _export_cmd(cpts: list, dry_run: bool = False) -> list:
    cmd = [
        sys.executable,
        str(WAREHOUSE_EXPORT_PY),
        "--project", SLUG,
        "--family", FAMILY,
        "--years", YEARS,
        "--cpt", ",".join(cpts),
        "--var-file", str(VAR_FILE),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_warehouse_export(cpts: list, dry_run: bool = False) -> None:
    cmd = _export_cmd(cpts, dry_run=dry_run)
    print(f"[01] Running warehouse export (read-only subprocess):")
    print("[01]   " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(WAREHOUSE_DIR), capture_output=True, text=True)
    for line in (result.stdout or "").rstrip().splitlines():
        print("[01]   [warehouse] " + line)
    if result.returncode != 0:
        print("[01]   [warehouse STDERR]")
        for line in (result.stderr or "").rstrip().splitlines():
            print("[01]   " + line)
        raise SystemExit(f"[01] warehouse export.py failed (exit {result.returncode})")

    # export.py logs to stdout and WARNS to stderr even on success. Partial-year-coverage
    # warnings are EXPECTED here: EMERGNCY/ELECTSURG stop at 2020, CASETYPE starts at
    # 2021, the ICD-9 fields stop at 2020 and the ICD-10 fields start at 2014. Those are
    # the documented era boundaries (.build_facts.md 3, CLAUDE.md 9.10), not errors.
    warn_lines = [ln for ln in (result.stderr or "").rstrip().splitlines() if ln.strip()]
    partial = [ln for ln in warn_lines if "partial year coverage" in ln]
    other = [ln for ln in warn_lines if "partial year coverage" not in ln]
    if partial:
        print(f"[01]   [warehouse] {len(partial)} 'partial year coverage' warning(s) "
              f"-- EXPECTED, not a failure (documented era boundaries).")
        for ln in partial[:3]:
            print("[01]     e.g. " + ln)
        if len(partial) > 3:
            print(f"[01]     ... and {len(partial) - 3} more of the same shape.")
    for ln in other:
        print("[01]   [warehouse] " + ln)
    if any("absent from view" in ln for ln in other):
        raise SystemExit(
            "[01] the exporter reported columns ABSENT from view 'main'. "
            "vars/acdf_discharge_timing.txt was verified complete; investigate before "
            "proceeding."
        )


# ---------------------------------------------------------------------------
# Bundle copy
# ---------------------------------------------------------------------------
def _warehouse_bundle_dir() -> pathlib.Path:
    return WAREHOUSE_DIR / "exports" / SLUG


def _project_bundle_dir() -> pathlib.Path:
    return PROJECT_EXPORTS_DIR / SLUG


def _copy_bundle() -> pathlib.Path:
    src_dir = _warehouse_bundle_dir()
    dst_dir = _project_bundle_dir()
    if not src_dir.exists():
        raise SystemExit(f"[01] warehouse bundle not found at {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in BUNDLE_FILES if not (src_dir / n).exists()]
    if missing:
        raise SystemExit(f"[01] warehouse bundle {src_dir} is missing {missing}")
    for name in BUNDLE_FILES:
        shutil.copy2(src_dir / name, dst_dir / name)      # overwrite = idempotent
    print(f"[01]   copied 6-file bundle -> {dst_dir}")
    print("[01]     " + ", ".join(BUNDLE_FILES))
    return dst_dir


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _scan_cohort_csv(csv_path: pathlib.Path):
    """One streaming pass over cohort.csv.

    Row counting goes through csv.reader, which honours RFC-4180 quoting, so an embedded
    newline in a free-text field (PRNCPTX) cannot inflate the count.

    The INOUT tally rides on this same pass (it is a level counter, not a second read of
    the file), and is checked in `_verify_bundle` below. A blank INOUT is counted under
    the key '' so the non-blank count is exact rather than inferred.
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for required in ("CPT", "AGE", "PUFYEAR"):
            if required not in header:
                raise SystemExit(f"[01] cohort.csv has no {required} column: {csv_path}")
        i_cpt = header.index("CPT")
        i_age = header.index("AGE")
        i_year = header.index("PUFYEAR")
        # INOUT is verified, not assumed: its absence is a hard failure below, so resolve
        # the index defensively and let `_verify_bundle` produce the message.
        i_inout = header.index(INOUT_COL) if INOUT_COL in header else None

        n_rows = 0
        by_cpt: Counter = Counter()
        by_year: Counter = Counter()
        by_inout: Counter = Counter()
        age_bad = 0
        age_missing = 0
        age_min, age_max = None, None
        for row in reader:
            n_rows += 1
            by_cpt[row[i_cpt].strip()] += 1
            by_year[row[i_year].strip()] += 1
            if i_inout is not None:
                by_inout[row[i_inout].strip()] += 1
            raw_age = row[i_age].strip()
            if raw_age == "":
                age_missing += 1
                continue
            try:
                age = int(float(raw_age))
            except ValueError:
                age_bad += 1
                continue
            if age < MIN_AGE:
                age_bad += 1
            age_min = age if age_min is None else min(age_min, age)
            age_max = age if age_max is None else max(age_max, age)
    return (header, n_rows, by_cpt, by_year, by_inout,
            age_bad, age_missing, age_min, age_max)


def _verify_inout(header: list, by_inout: Counter, n_rows: int) -> None:
    """HARD verification of the one column this project adds to the parent's export.

    Three checks, in order:
      1. the column is in the header at all;
      2. it is populated on the identified cohort (100% non-blank is the expectation; a
         blank is reported with its exact count rather than swallowed);
      3. it carries ONLY 'Inpatient' and 'Outpatient'.

    Check 3 is a STOP CONDITION. `05_inout.py` cross-tabulates this column against true
    postoperative length of stay as a 2x2, so a third level would either be dropped
    silently or land in the wrong cell, and either way the sensitivity, specificity, PPV,
    NPV and kappa in Table 4 would be computed on a corrupted table that still looks
    perfectly plausible on the page. The failure mode is a wrong number, not an error, so
    it has to raise here.
    """
    if INOUT_COL not in header:
        raise SystemExit(
            f"[01] cohort.csv has no {INOUT_COL} column. It is listed in "
            f"{VAR_FILE.name} and is REQUIRED by this project (PLAN.md section 6.4). "
            f"If the exporter reported it absent from view 'main', stop and investigate."
        )

    blank = by_inout.get("", 0)
    non_blank = n_rows - blank
    pct = 100.0 * non_blank / n_rows if n_rows else 0.0
    print(f"[01]   {INOUT_COL} level counts (identified cohort, n = {n_rows:,}):")
    for level in sorted(by_inout, key=lambda k: (-by_inout[k], k)):
        shown = level if level != "" else "(blank)"
        print(f"[01]     {shown:<14} {by_inout[level]:>10,}  "
              f"({100.0 * by_inout[level] / n_rows:5.2f}%)")

    if blank:
        print(f"[01]   NOTE  {INOUT_COL} is NOT 100% populated: {non_blank:,} of "
              f"{n_rows:,} rows non-blank ({pct:.4f}%), {blank:,} blank. "
              f"02_build_cohort.py must keep the recode NULL-safe and 05_inout.py must "
              f"state this denominator.")
    else:
        print(f"[01]   OK  {INOUT_COL} is 100% non-blank: {non_blank:,} / {n_rows:,}.")

    unexpected = sorted(lv for lv in by_inout if lv != "" and lv not in INOUT_LEVELS)
    if unexpected:
        raise SystemExit(
            f"[01] {INOUT_COL} carries unexpected level(s) {unexpected}. Only "
            f"{list(INOUT_LEVELS)} are known (.build_facts.md section 3, measured "
            f"read-only on the curated `main` view for 2012-2024).\n"
            f"     This is a STOP CONDITION: an unmapped level would silently corrupt the "
            f"2x2 in PLAN.md section 6.4 and every statistic derived from it (sensitivity, "
            f"specificity, PPV, NPV, kappa) without raising anywhere downstream.\n"
            f"     Do not map the new level by guesswork. Establish what it means against "
            f"the PUF guides first, then extend INOUT_LEVELS here and the explicit map in "
            f"02_build_cohort.py together."
        )
    print(f"[01]   OK  {INOUT_COL} carries only {list(INOUT_LEVELS)}; no unmapped level.")


def _verify_bundle(cpts: list, bundle_dir: pathlib.Path) -> dict:
    """HARD verification. Any mismatch raises SystemExit."""
    csv_path = bundle_dir / "cohort.csv"
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    manifest_n = manifest.get("row_count")
    manifest_cols = manifest.get("n_columns")

    (header, csv_n, by_cpt, by_year, by_inout,
     age_bad, age_missing, age_min, age_max) = _scan_cohort_csv(csv_path)

    print(f"[01]   manifest.json  row_count = {manifest_n:,}   n_columns = {manifest_cols}")
    print(f"[01]   cohort.csv     data rows = {csv_n:,} (csv.reader)   "
          f"header fields = {len(header)}")

    if manifest_n != csv_n:
        raise SystemExit(
            f"[01] manifest row_count ({manifest_n:,}) != independent cohort.csv count "
            f"({csv_n:,}). The bundle is internally inconsistent."
        )

    if csv_n != EXPECTED_N:
        raise SystemExit(
            f"[01] N = {csv_n:,} but cpt_acdf.COHORT_LADDER_N['identified'] = "
            f"{EXPECTED_N:,} (delta {csv_n - EXPECTED_N:+,}).\n"
            f"     This is warehouse-verified ground truth, NOT an approximation, and it\n"
            f"     is Gate 1 (CLAUDE.md 8.3): adding INOUT to the variable list must not\n"
            f"     move a single row count.\n"
            f"     CLAUDE.md 8.5: a mismatch is a STOP CONDITION, not a number to\n"
            f"     overwrite. Investigate with:\n"
            f"       {' '.join(_export_cmd(cpts, dry_run=True))}"
        )
    print(f"[01]   OK  N = {csv_n:,} exactly matches cpt_acdf.COHORT_LADDER_N"
          f"['identified'].")

    if len(header) != EXPECTED_COLS or manifest_cols != EXPECTED_COLS:
        raise SystemExit(
            f"[01] expected {EXPECTED_COLS} columns ({N_REQUESTED_COLS} requested + "
            f"{len(AUTO_INJECTED_COLS)} auto-injected), got header={len(header)} "
            f"manifest={manifest_cols}."
        )
    missing_ident = [c for c in AUTO_INJECTED_COLS if c not in header]
    if missing_ident:
        raise SystemExit(f"[01] auto-injected column(s) {missing_ident} absent from header")
    dupe_cols = sorted({c for c in header if header.count(c) > 1})
    if dupe_cols:
        raise SystemExit(f"[01] duplicate column(s) in header: {dupe_cols}")
    requested = _read_var_file(VAR_FILE)
    absent = [c for c in requested if c not in header]
    if absent:
        raise SystemExit(f"[01] requested column(s) missing from the bundle: {absent}")
    print(f"[01]   OK  {EXPECTED_COLS} columns; all {len(AUTO_INJECTED_COLS)} identity "
          f"columns and all {len(requested)} requested columns present; no duplicates.")

    # --- filters actually applied -------------------------------------------
    filters = manifest.get("filters", {})
    if sorted(filters.get("cpt") or []) != cpts:
        raise SystemExit(
            f"[01] manifest CPT filter {filters.get('cpt')} != the {len(cpts)} codes "
            f"from cpt_acdf.PRINCIPAL_CPTS."
        )
    want_years = list(range(2012, 2025))
    if list(filters.get("years") or []) != want_years:
        raise SystemExit(
            f"[01] manifest year filter is {filters.get('years')}, expected {want_years}."
        )
    if filters.get("where"):
        raise SystemExit(
            f"[01] an unexpected --where predicate is in the spec: "
            f"{filters.get('where')!r}. This export must be the unfiltered identified "
            f"cohort; every eligibility rule belongs to 02_build_cohort.py."
        )
    print(f"[01]   OK  filters: family={manifest.get('family')}  years=2012-2024  "
          f"cpt={len(cpts)} codes  where=None")

    # --- no CPT outside the requested set ------------------------------------
    stray = sorted(set(by_cpt) - set(cpts))
    if stray:
        raise SystemExit(f"[01] rows with unrequested principal CPT {stray}")

    # --- the age criterion is non-binding, and we PROVE it rather than assume -
    if age_missing or age_bad:
        raise SystemExit(
            f"[01] AGE is not clean: {age_missing:,} missing, {age_bad:,} below "
            f"{MIN_AGE} or unparseable. The export carries no age --where, so "
            f"02_build_cohort.py would have to drop them; that would make the "
            f"identified rung differ from {EXPECTED_N:,}. Investigate."
        )
    print(f"[01]   OK  AGE clean: 0 missing, 0 under {MIN_AGE}; range "
          f"{age_min}-{age_max} (90 is the NSQIP top-coded '90+' token).")

    if sorted(int(y) for y in by_year) != want_years:
        raise SystemExit(
            f"[01] PUFYEAR values in the bundle are {sorted(by_year)}, expected "
            f"{want_years}."
        )

    # --- the column this project adds ----------------------------------------
    _verify_inout(header, by_inout, csv_n)

    return {"n": csv_n, "header": header, "by_cpt": by_cpt, "by_year": by_year,
            "by_inout": by_inout, "manifest": manifest,
            "age_min": age_min, "age_max": age_max}


def _print_breakdown(cpts: list, res: dict) -> None:
    by_cpt: Counter = res["by_cpt"]
    by_year: Counter = res["by_year"]
    print("[01]   per-CPT composition (identified cohort):")
    print(f"[01]     {'CPT':<7} {'identified':>11}")
    print(f"[01]     {'-' * 7} {'-' * 11}")
    for code in sorted(by_cpt, key=lambda c: (-by_cpt[c], c)):
        print(f"[01]     {code:<7} {by_cpt[code]:>11,}")
    print(f"[01]     {'TOTAL':<7} {sum(by_cpt.values()):>11,}")

    print("[01]   per-year composition (identified cohort):")
    for y in sorted(by_year, key=lambda v: int(v)):
        print(f"[01]     {y}  {by_year[y]:>8,}")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def _existing_bundle_usable(cpts: list) -> bool:
    """True iff the project bundle is present, complete and the right shape.

    Cheap pre-check only (file presence + manifest); the full verification runs after
    either branch, so a bundle reused here is still verified to the row.
    """
    d = _project_bundle_dir()
    if not all((d / n).exists() for n in BUNDLE_FILES):
        return False
    try:
        m = json.loads((d / "manifest.json").read_text())
    except Exception:
        return False
    filters = m.get("filters", {})
    return (
        m.get("row_count") == EXPECTED_N
        and m.get("n_columns") == EXPECTED_COLS
        and m.get("family") == FAMILY
        and list(filters.get("years") or []) == list(range(2012, 2025))
        and sorted(filters.get("cpt") or []) == cpts
        and not filters.get("where")
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="01_export_cohort.py",
        description="Export the identified ACDF cohort from the NSQIP DuckDB warehouse "
                    "(read-only subprocess).",
    )
    ap.add_argument("--force", action="store_true",
                    help="re-run the warehouse export even if a valid bundle exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exporter's resolved SQL and exit; writes nothing")
    args = ap.parse_args(argv)

    cpts = _principal_cpts()

    print("=" * 78)
    print("[01] ACDF discharge timing -- identified cohort export")
    print(f"[01] warehouse : {WAREHOUSE_DIR}  (READ-ONLY)")
    print(f"[01] project   : {PROJECT_DIR}")
    print(f"[01] slug={SLUG}  family={FAMILY}  years={YEARS}  cpt={len(cpts)} codes")
    print(f"[01] cpt = {','.join(cpts)}   (from cpt_acdf.PRINCIPAL_CPTS)")
    print("=" * 78)

    _preflight()

    if args.dry_run:
        _run_warehouse_export(cpts, dry_run=True)
        print("[01] DRY RUN complete; nothing written.")
        return 0

    if not args.force and _existing_bundle_usable(cpts):
        print(f"[01]   bundle already present and consistent at {_project_bundle_dir()}")
        print("[01]   skipping the warehouse export; re-verifying it anyway "
              "(use --force to re-export).")
    else:
        _run_warehouse_export(cpts)
        _copy_bundle()

    res = _verify_bundle(cpts, _project_bundle_dir())
    _print_breakdown(cpts, res)

    print("")
    print("=" * 78)
    print("[01] SUMMARY")
    print(f"[01]   {_project_bundle_dir()}")
    print(f"[01]   identified {res['n']:,} rows  x  {len(res['header'])} cols")
    print(f"[01]   {INOUT_COL}: " + "  ".join(
        f"{lv}={res['by_inout'].get(lv, 0):,}" for lv in INOUT_LEVELS))
    print(f"[01]   next: python3 02_build_cohort.py   "
          f"(read the CSV with keep_default_na=False -- CLAUDE.md 9.8)")
    print(f"[01] DONE. Row count exact, column count {EXPECTED_COLS}.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
