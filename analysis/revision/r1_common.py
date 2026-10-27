#!/usr/bin/env python3
"""r1_common.py - shared loader and helpers for the additional analyses in this folder.

Every script in analysis/revision/ imports THIS module. It loads the original estimator
(06_window.py), the original length-of-stay and trend code (05_inout.py) and the label layer
(variable_dict.py) by importlib from the analysis folder, reads the frozen analytic parquet
READ-ONLY, and offers the small set of helpers the analyses share. The original scripts are never
edited and never rerun as __main__; sys.dont_write_bytecode is set below BEFORE the first import
of an original script, so no .pyc is written next to them. "v1" in this folder means the original
pipeline in analysis/ (release v1.0.0).

WHERE THINGS LIVE (three environment variables, all optional)

  ACDF_ANALYSIS_DIR  the folder that holds 06_window.py, 05_inout.py, variable_dict.py and
                     derived/cohort_analytic.parquet (and, after steps 03 and 06 have been run,
                     tables-csv/ and outputs/). Default: the repository's analysis/ folder.
                     It is only ever READ.
  ACDF_RESULTS_DIR   where every output of these scripts is written. Default: results/revision/
                     of the repository. The scripts refuse to write anywhere else.
  ACDF_FIGURES       set to 0 to skip the PNG and PDF figures (default: write them).

HOW A SCRIPT SHOULD START (the first two lines keep this folder free of .pyc files too):

    import sys; sys.dont_write_bytecode = True
    from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
    import r1_common as R1
    import numpy as np
    R1.seed_all()
    df = R1.load_cohort()

and run it as   PYTHONDONTWRITEBYTECODE=1 python3 a1_something.py

PUBLIC API (everything else in this file is private)

  constants    REPO_ROOT ANALYSIS_DIR RESULTS_DIR ORIG_TABLES ORIG_OUTPUTS TABLES_OUT
               FRAGMENTS_OUT FIGURES_OUT MERGED_JSON FIGURES COHORT_PARQUET
               EXPECTED_COHORT_N REQUIRED_COLUMNS DAYS UNREACHED_DAY YEARS COVID_YEAR
               TREND_COVARIATES R1_THRESHOLDS suppressed_text
               palette: BLUE ORANGE BLUE_STEP INK INK_MUTED RULE SERIES_STYLE
                        THRESHOLD_LINESTYLE OVERALL_LINESTYLE FIG_SIZE_FOUR_PANEL
                        FOUR_PANEL_ADJUST V1_FIGURE_FONT_PT
  v1 modules   VD (variable_dict), W (06_window), IO (05_inout), load_original_module(filename)
  data         load_cohort(), make_events(df, event_col, pod_col, ...), Events,
               sameday_counts_by_year(df), build_trend_frame(df, outcome), fit_year_logit(...)
  estimation   fit_overall_and_strata(df, events, ...), StratumFit, BootBlock,
               fit_strata_v1(df, events, ...), threshold_day(curve, t), day_or_none(day),
               stays_below_from(curve, t, day), index_by_slug(fits)
  thresholds   threshold_text(t), threshold_slug(t)
  output       write_fragment(path, results, prefixes), write_csv_table(path, header, rows),
               assert_clean_text(text), style_figure(...), plot_step_curve(...),
               save_png(fig, path, also_pdf=True)
  seeds        seed_all()

RNG-ORDER CONVENTION (it is part of the result; do not break it)
  v1 seeds the global NumPy stream once (np.random.seed(VD.SEED), VD.SEED = 20260731) and
  then visits the strata in W.build_strata(df) order: the overall row, then the four locked
  stratifiers in VD.STRATIFIERS order with the residual sex level last. For every stratum
  that is NOT suppressed it makes exactly one draw
  np.random.multinomial(n, probs, size=1000) through W.bootstrap_cells. A suppressed stratum
  (fewer than VD.MIN_STRATUM_EVENTS = 10 events of the outcome being fitted) makes NO draw.
  Nothing else touches the stream between draws. fit_overall_and_strata() follows exactly
  this order and, by default, RESEEDS with VD.SEED at the top of each call, so the draws for
  one outcome never depend on which outcomes were fitted earlier in the same script, and the
  primary composite reproduces v1's own bootstrap draws bit for bit (self_test() checks it).
  Pass seed=None to continue the current stream instead. Do not reorder, parallelise or skip
  strata inside a call, and do not call np.random.* between strata of one call.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # BEFORE the first import of an original script: no .pyc next to them

import argparse
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ===========================================================================
# 1. PATHS AND PINNED CONSTANTS
# ===========================================================================


def _env_dir(name: str, default: Path) -> Path:
    """The directory named by environment variable `name`, else `default` (both resolved)."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


REPO_ROOT = Path(__file__).resolve().parents[2]            # analysis/revision/r1_common.py -> repository root

# The analysis folder holds the ORIGINAL scripts and the frozen cohort. It is only ever read.
ANALYSIS_DIR = _env_dir("ACDF_ANALYSIS_DIR", REPO_ROOT / "analysis")
ORIG_TABLES = ANALYSIS_DIR / "tables-csv"                  # written by the original steps 03 to 06
ORIG_OUTPUTS = ANALYSIS_DIR / "outputs"                    # results_NN.json fragments of steps 02 to 06
COHORT_PARQUET = ANALYSIS_DIR / "derived" / "cohort_analytic.parquet"

# Every output of these scripts goes under RESULTS_DIR, and nowhere else.
RESULTS_DIR = _env_dir("ACDF_RESULTS_DIR", REPO_ROOT / "results" / "revision")
TABLES_OUT = RESULTS_DIR / "tables-csv"
FRAGMENTS_OUT = RESULTS_DIR / "fragments"
FIGURES_OUT = RESULTS_DIR / "figures"
MERGED_JSON = RESULTS_DIR / "results_r1.json"

# ACDF_FIGURES=0 skips the PNG and PDF figures; every number and table is unaffected.
FIGURES = os.environ.get("ACDF_FIGURES", "1").strip().lower() not in ("0", "false", "no", "off")

EXPECTED_COHORT_N = 112_642

if not (ANALYSIS_DIR / "06_window.py").exists() or not (ANALYSIS_DIR / "variable_dict.py").exists():
    raise RuntimeError(f"06_window.py and variable_dict.py were not found in {ANALYSIS_DIR}; "
                       f"set ACDF_ANALYSIS_DIR to the folder that holds the original scripts")
if RESULTS_DIR == ANALYSIS_DIR or ANALYSIS_DIR in RESULTS_DIR.parents:
    raise RuntimeError(f"ACDF_RESULTS_DIR ({RESULTS_DIR}) must not be inside the analysis folder "
                       f"({ANALYSIS_DIR}), which is read only")

# Columns the additional analyses read from the parquet. load_cohort() asserts them.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "CASE_UID", "OPERYR", "AGE_CAT", "ASA_CAT", "LEVELS_CAT", "MALE", "SEX",
    "AGE_PER10", "BMI_PER5", "ASA_GE3",
    "WINDOW_COMP", "POD_WINDOW", "HEMATOMA", "HEMATOMA_POD",
    "AIRWAY_COMP", "POD_AIRWAY", "REINTUBATION", "VENT48", "REOP_TRACH",
    "MORT30", "POD_DEATH", "LOS_POSTOP", "STILLINHOSP", "SAMEDAY_DISCH",
    "DISCHDEST", "RETORPODAYS",
)

suppressed_text = "Suppressed"      # the one word a withheld table cell carries (matches v1)

# ===========================================================================
# 2. THE v1 MODULES (importlib, read-only)
# ===========================================================================

_V1_MODULES: Dict[str, Any] = {}


def _module_name(filename: str) -> str:
    stem = Path(filename).stem
    m = re.match(r"^\d\d_(.+)$", stem)
    return f"v1_{m.group(1)}" if m else stem


def load_original_module(filename: str):
    """Import one original source file by path and return the module (cached).

    `06_window.py` becomes module `v1_window`, `05_inout.py` becomes `v1_inout`, and a plain
    name such as `variable_dict.py` keeps its own name so that the original `import variable_dict`
    statements resolve to the SAME module object. ANALYSIS_DIR is put on sys.path so those imports
    work. Importing writes nothing under ANALYSIS_DIR because dont_write_bytecode is set and
    05/06 only write inside main(). Never call an original main().
    """
    name = _module_name(filename)
    if name in _V1_MODULES:
        return _V1_MODULES[name]
    path = ANALYSIS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses in 06_window.py resolve their module here
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _V1_MODULES[name] = mod
    return mod


VD = load_original_module("variable_dict.py")     # labels, SEED, STRATIFIERS, formatters
W = load_original_module("06_window.py")          # the estimator (Aalen-Johansen, strata, tables)
IO = load_original_module("05_inout.py")          # length of stay, trend model (build_design)

DAYS = W.DAYS                       # 31: postoperative days 0 to 30
UNREACHED_DAY = W.UNREACHED_DAY     # 31: sentinel for "threshold never crossed inside 30 days"
YEARS: Tuple[int, ...] = tuple(IO.YEARS)       # 2012 to 2024
COVID_YEAR = IO.COVID_YEAR                     # 2020

# Covariates of the v1 adjusted same-day trend model (05_inout.py main()).
TREND_COVARIATES: Tuple[str, ...] = ("AGE_PER10", "MALE", "BMI_PER5", "ASA_GE3", "LEVELS_CAT")

# Thresholds these analyses may use. The first three are v1's VD.RISK_THRESHOLDS (1%, 0.5%, 0.25%);
# 0.10% is added here, and v1's VD.threshold_label() raises on it, so use threshold_text() and
# threshold_slug() below instead.
R1_THRESHOLDS: Tuple[float, ...] = (0.01, 0.005, 0.0025, 0.001)


class R1Error(RuntimeError):
    """Raised for a violated data or output contract. Never caught inside this file."""


# ===========================================================================
# 3. THE COHORT AND THE EVENTS
# ===========================================================================


def load_cohort() -> pd.DataFrame:
    """Read derived/cohort_analytic.parquet from the analysis folder (READ ONLY) and return the DataFrame.

    Asserts 112,642 rows, a unique CASE_UID, and every column in REQUIRED_COLUMNS. The frame
    stays in memory; no analysis may write it, or any patient-level extract, anywhere.
    """
    if not COHORT_PARQUET.exists():
        raise R1Error(f"missing {COHORT_PARQUET}; run steps 01 and 02 first, or set ACDF_ANALYSIS_DIR")
    df = pd.read_parquet(COHORT_PARQUET)
    if len(df) != EXPECTED_COHORT_N:
        raise R1Error(f"analytic parquet has {len(df):,} rows, expected {EXPECTED_COHORT_N:,}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise R1Error(f"columns missing from the analytic parquet: {missing}")
    if not df["CASE_UID"].is_unique:
        raise R1Error("CASE_UID is not unique in the analytic parquet")
    return df


@dataclass
class Events:
    """One outcome's per-patient event arrays, aligned with the rows of the cohort frame.

    All arrays have len(df). Days are floats holding integers 0 to 30, NaN where not applicable.
      is_event      bool   the primary event counts for this patient: flag set AND timed (see
                           make_events for untimed events) AND the patient is not a death-first
                           patient. A patient whose death day is STRICTLY BEFORE their primary
                           event day enters ONLY as the competing event: is_event is False and
                           is_competing is True, so no patient is ever in both e1 and e2. n_events
                           can therefore be smaller than n_event_flagged by n_death_strictly_first
                           (plus any untimed events dropped); that count is 0 for the composite,
                           hematoma and airway outcomes (self_test() asserts it).
      pod_event     float  postoperative day of the primary event, NaN where not is_event
      is_death      bool   MORT30 == 1 (death within 30 days)
      pod_death     float  postoperative day of death, NaN where not is_death
      is_competing  bool   death enters the estimator as the competing event (death strictly before
                           the primary event, or no counted primary event; a same-day tie goes to
                           the primary event). Mutually exclusive with is_event.
      exit_day      float  day the patient leaves the risk set: event day, else competing death
                           day, else +inf (reaches day 30 event free and alive)
      los           float  LOS_POSTOP, postoperative length of stay in days, NaN where not recorded
      n_untimed_events  int  flag set but no day recorded (see make_events for the treatment)
    The remaining fields are counts for reporting. n_both counts patients with a timed primary
    event flag AND a death, BEFORE the death-first rule, so n_both = n_death_strictly_first +
    n_tie_same_day + (patients whose event came strictly before their death).
    """
    event_col: str
    pod_col: str
    untimed: str
    is_event: np.ndarray
    pod_event: np.ndarray
    is_death: np.ndarray
    pod_death: np.ndarray
    is_competing: np.ndarray
    exit_day: np.ndarray
    los: np.ndarray
    n_untimed_events: int
    n_event_flagged: int = 0          # rows with the event flag set (timed or not)
    n_events: int = 0                 # is_event.sum() (after the death-first rule)
    n_death: int = 0                  # is_death.sum()
    n_competing: int = 0              # is_competing.sum()
    n_both: int = 0                   # timed event flag & death, before the death-first rule
    n_death_strictly_first: int = 0   # both, death day before the event day: competing ONLY
    n_tie_same_day: int = 0           # both, same day (the primary event takes the tie)


def make_events(df: pd.DataFrame, event_col: str, pod_col: str, untimed: str = "raise",
                untimed_day: Optional[int] = None,
                death_col: str = "MORT30", death_pod_col: str = "POD_DEATH") -> Events:
    """Build the Events arrays for one primary event flag and its postoperative-day column.

    Generalises 06_window.py main() lines 2397 to 2431 to any flag, with death (MORT30,
    POD_DEATH) as the competing event. With event_col="WINDOW_COMP", pod_col="POD_WINDOW" it
    reproduces main() exactly (1,015 events, 213 deaths, 119 deaths entering as competing).

    TIE RULE (same as main()): a patient with both a primary event and a death is a PRIMARY
    event when the event day is on or before the death day; death is the competing event only
    when it comes strictly first or the patient has no timed primary event.

    DEATH-FIRST RULE (one deliberate difference from main()'s literal arrays). A patient whose
    death day is STRICTLY BEFORE their primary-event day "enters as the competing event" (v1's
    own stated intent, 06_window.py main() lines 2432 to 2434; v1 itself is unchanged). main()
    keeps is_event True for that
    patient while is_competing is also True, which would count them in both e1 and e2; this
    function sets is_event False (pod_event NaN) for them and keeps is_competing True and
    n_death_strictly_first. So exactly one of is_event / is_competing holds for every such
    patient, and n_events can be smaller than the flagged event count by n_death_strictly_first
    (plus untimed events dropped, see below). Today n_death_strictly_first is 0 for the
    composite, hematoma (untimed="drop") and airway outcomes, so nothing changes numerically and
    1,015 / 540 / 608 still hold; self_test() asserts it.

    UNTIMED EVENTS. An untimed event is a patient whose flag is 1 but whose day column is
    missing (the value is never imputed and never given a boundary value, as in the original
    cohort build). The parquet holds exactly one for HEMATOMA: HEMATOMA == 1,
    HEMATOMA_POD missing because the raw RETORPODAYS is itself missing (not a range-rule
    casualty), and that patient also has AIRWAY_COMP == 1 with POD_AIRWAY = 1, so the
    composite carries them on day 1 (POD_WINDOW = 1) and the composite never sees an untimed
    event. How `untimed` treats it:
      "raise"  (default, used for the composite) raise R1Error if any event is untimed.
      "drop"   the patient is NOT an event (is_event False, pod_event NaN) but STAYS in the
               denominator of every stratum they belong to, as an event-free patient who
               leaves the risk set only by death (if they die) or by reaching day 30. The
               count is reported in n_untimed_events. This is NOT conservative for a safety
               claim: it removes one event from the numerator, so residual risk is biased
               downward by at most 1 event in 541 (0.18 percent relative for hematoma), and a
               threshold day can only move earlier or stay put. Say so wherever it is used.
      "assign" SENSITIVITY ONLY. The untimed events are placed on `untimed_day` (required),
               e.g. 30 to give the most conservative bound (the event then adds to the residual
               risk on every day 0 to 30) or 1 to give the day their airway event was recorded.
    n_untimed_events is filled in every mode. Nothing is silently clipped or dropped: an event
    day that is not an integer in 0 to 30, or a day present on a row whose flag is not set,
    raises R1Error (the 02 build froze the handling rule; no step may clip locally).
    """
    if untimed not in ("raise", "drop", "assign"):
        raise ValueError("untimed must be 'raise', 'drop' or 'assign'")
    if untimed == "assign" and untimed_day is None:
        raise ValueError("untimed='assign' requires untimed_day")
    if untimed == "assign" and not (isinstance(untimed_day, (int, np.integer))
                                    and 0 <= int(untimed_day) <= VD.FOLLOWUP_DAYS):
        raise ValueError("untimed_day must be an integer 0 to 30")

    for col in (event_col, pod_col, death_col, death_pod_col, "LOS_POSTOP"):
        if col not in df.columns:
            raise R1Error(f"column {col} is missing from the cohort frame")
    if df[event_col].isna().any() or df[death_col].isna().any():
        raise R1Error(f"{event_col} or {death_col} carries missing flags; the analytic "
                      f"parquet is expected to be complete on both")

    flagged = (df[event_col] == 1).to_numpy(dtype=bool)
    pod_raw = df[pod_col].to_numpy(dtype=float, na_value=np.nan)
    is_death = (df[death_col] == 1).to_numpy(dtype=bool)
    pod_death = df[death_pod_col].to_numpy(dtype=float, na_value=np.nan)
    los = df["LOS_POSTOP"].to_numpy(dtype=float, na_value=np.nan)

    if np.any(np.isfinite(pod_raw[~flagged])):
        raise R1Error(f"{pod_col} is populated on a row whose {event_col} flag is not set")
    if not np.all(np.isfinite(pod_death[is_death])):
        raise R1Error(f"a recorded death carries no {death_pod_col}")

    untimed_mask = flagged & ~np.isfinite(pod_raw)
    n_untimed = int(untimed_mask.sum())
    pod_event = pod_raw.copy()
    if n_untimed:
        if untimed == "raise":
            raise R1Error(f"{n_untimed} {event_col} event(s) carry no {pod_col}. Use "
                          f"untimed='drop' (documented in make_events) or report the mismatch; "
                          f"never impute silently.")
        if untimed == "drop":
            is_event = flagged & ~untimed_mask
        else:
            is_event = flagged.copy()
            pod_event[untimed_mask] = float(untimed_day)
    else:
        is_event = flagged.copy()
    pod_event = np.where(is_event, pod_event, np.nan)

    def _clean(days: np.ndarray, mask: np.ndarray) -> bool:
        d = days[mask]
        return bool(np.all((d >= 0) & (d <= VD.FOLLOWUP_DAYS) & (d % 1 == 0)))

    if not (_clean(pod_event, is_event) and _clean(pod_death, is_death)):
        raise R1Error("a postoperative day is not an integer in 0 to 30; 02 froze the "
                      "handling rule and no step may clip or drop locally")

    both = is_event & is_death
    death_first = both & (pod_death < pod_event)
    n_death_first = int(death_first.sum())
    n_tie = int(np.sum(both & (pod_death == pod_event)))
    is_competing = is_death & ~(is_event & (pod_event <= pod_death))
    # DEATH-FIRST RULE: a patient who dies strictly before their event day is the competing event
    # ONLY, never also a primary event (see the make_events docstring). `both` and n_death_first
    # were taken above, before this step, so n_both still counts them.
    is_event = is_event & ~death_first
    pod_event = np.where(is_event, pod_event, np.nan)

    exit_day = np.full(len(df), np.inf, dtype=float)
    exit_day[is_event] = pod_event[is_event]
    exit_day[is_competing] = pod_death[is_competing]

    return Events(
        event_col=event_col, pod_col=pod_col, untimed=untimed,
        is_event=is_event, pod_event=pod_event, is_death=is_death, pod_death=pod_death,
        is_competing=is_competing, exit_day=exit_day, los=los,
        n_untimed_events=n_untimed, n_event_flagged=int(flagged.sum()),
        n_events=int(is_event.sum()), n_death=int(is_death.sum()),
        n_competing=int(is_competing.sum()), n_both=int(both.sum()),
        n_death_strictly_first=n_death_first, n_tie_same_day=n_tie)


# ===========================================================================
# 4. ESTIMATION: overall plus the four locked stratifiers
# ===========================================================================


@dataclass
class BootBlock:
    """Within-stratum multinomial bootstrap (1,000 replicates by default), B = replicates.

    e1, e2   (B, 31) resampled event and competing-death counts per postoperative day
    cif      (B, 32) Aalen-Johansen cumulative incidence at the START of day d (d = 0..30), col 31 = end of day 30
    surv     (B, 32) probability event free AND alive, same indexing
    curve    (B, 31) conditional risk of a subsequent event, column D = risk on day D or later
                     given event free and alive at the start of day D (column 0 = day-30 cumulative incidence)
    lo, hi   (31,)   2.5th and 97.5th percentile of `curve` (W.percentile_ci), the v1 interval
    cif30_lo, cif30_hi  percentile interval of the day-30 cumulative incidence (proportion)
    """
    e1: np.ndarray
    e2: np.ndarray
    cif: np.ndarray
    surv: np.ndarray
    curve: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    cif30_lo: float
    cif30_hi: float


@dataclass
class StratumFit:
    """Counts, Aalen-Johansen point estimates and bootstrap for one stratum (all in proportions).

    stratum      the W.Stratum (slug, order, stratifier, level, column, raw_level, ...). Take display
                 text from stratum.level, never from the slug (slug 'sex.not-recorded' is the 35
                 NON-BINARY patients, label 'Non-binary', not unrecorded sex)
    mask         bool (len(df),) membership, IN MEMORY ONLY (never write it)
    n, events, competing   patients, primary events, deaths entering as the competing event
    suppressed   True when events < VD.MIN_STRATUM_EVENTS (all arrays below are then None and
                 NO bootstrap draw was made); the estimate must be withheld, counts still shown
    e1, e2       (31,) events and competing deaths ON each postoperative day
    at_risk      (31,) patients event free and alive at the START of day d (the conditioning set)
    cif          (32,) cumulative incidence at the start of day d, col 31 = terminal (end of day 30)
    surv         (32,) probability event free and alive, same indexing
    curve        (31,) conditional risk at day d (W.conditional_risk_aj); curve[0] == cif[31]
    cif30        float day-30 cumulative incidence (proportion)
    boot         BootBlock or None (bootstrap=False, or suppressed)
    """
    stratum: Any
    mask: np.ndarray
    n: int
    events: int
    competing: int
    suppressed: bool
    reason: str = ""
    e1: Optional[np.ndarray] = None
    e2: Optional[np.ndarray] = None
    at_risk: Optional[np.ndarray] = None
    cif: Optional[np.ndarray] = None
    surv: Optional[np.ndarray] = None
    curve: Optional[np.ndarray] = None
    cif30: float = float("nan")
    boot: Optional[BootBlock] = field(default=None, repr=False)

    @property
    def slug(self) -> str:
        """Fragment-key segment of the stratum, e.g. 'overall', 'age.18-49', 'sex.not-recorded'.

        WARNING: the slug 'sex.not-recorded' is the 35 NON-BINARY patients (display label
        'Non-binary'), NOT patients with unrecorded sex. A slug is a key, never prose: take every
        word a reader sees from `stratum.level` (or VD.pretty_level), never from the slug.
        """
        return self.stratum.slug

    def threshold_day(self, threshold: float) -> int:
        """First day the point-estimate conditional risk is strictly below `threshold` (31 = never).

        This is the FIRST crossing only. The conditional risk is not guaranteed monotone, so the
        curve can rise above `threshold` again on a later day and this still returns the first
        day. v1 reports that separately as W.stays_below (06_window.py line 314, the "remains
        below" column of Table 5, key window.<stratum>.<threshold>.stays_below). Any analysis
        that reports a threshold day MUST also report "stays below through day 30" beside it:
        use stays_below_from(self.curve, threshold, day).
        """
        if self.suppressed:
            raise R1Error(f"stratum {self.slug} is suppressed")
        return int(W.first_day_below(self.curve[None, :], threshold)[0])

    def boot_threshold_days(self, threshold: float) -> np.ndarray:
        """(B,) first day below `threshold` in each bootstrap replicate (31 = never)."""
        if self.boot is None:
            raise R1Error(f"stratum {self.slug} has no bootstrap block")
        return W.first_day_below(self.boot.curve, threshold)


def fit_overall_and_strata(df: pd.DataFrame, events: Events, bootstrap: bool = True,
                           replicates: int = W.BOOTSTRAP_REPLICATES,
                           seed: Optional[int] = VD.SEED) -> List[StratumFit]:
    """Aalen-Johansen fit (death competing) for the overall row and every stratum, in v1 order.

    Strata come from W.build_strata(df): overall, age group (4 levels), ASA class (2), fusion
    levels (3), sex (male, female, and the 35 non-binary patients as the residual level), in
    that order, 13 strata in all. Returns a list of StratumFit in that order. The four
    stratifiers are asserted to partition the cohort.

    WARNING on the residual sex level: its slug is 'sex.not-recorded' but it is the 35
    NON-BINARY patients (display label 'Non-binary'), NOT patients with unrecorded sex. Take
    every word a reader sees from `fit.stratum.level`, never from the slug.

    Uses the v1 functions W.counts_by_day, W.aalen_johansen, W.conditional_risk_aj,
    W.bootstrap_cells, W.percentile_ci. For the primary composite its curves, intervals and
    bootstrap threshold-day draws equal those of W.fit_stratum exactly (self_test() checks it).
    Works for any Events (hematoma, airway, ...); only the suppression rule differs by event.

    seed=VD.SEED (default) reseeds the global stream first, which is what reproduces v1's draws
    (see the module docstring for the RNG-order convention). seed=None continues the current
    stream. bootstrap=False draws nothing. A suppressed stratum (fewer than 10 events) makes
    no draw and carries no estimates.
    """
    if seed is not None:
        np.random.seed(seed)
    fits: List[StratumFit] = []
    for stratum, mask in W.build_strata(df):
        n = int(mask.sum())
        ev_mask = mask & events.is_event
        cp_mask = mask & events.is_competing
        n_ev = int(ev_mask.sum())
        n_cp = int(cp_mask.sum())
        if n_ev < VD.MIN_STRATUM_EVENTS:
            fits.append(StratumFit(stratum, mask, n, n_ev, n_cp, True,
                                   f"fewer than {VD.MIN_STRATUM_EVENTS} events ({n_ev} observed)"))
            continue
        e1, e2 = W.counts_by_day(events.pod_event[ev_mask], events.pod_death[cp_mask])
        cif, surv, at_risk = W.aalen_johansen(e1[None, :], e2[None, :], np.array([n], dtype=float))
        curve = W.conditional_risk_aj(cif, surv)[0]
        fit = StratumFit(stratum, mask, n, n_ev, n_cp, False, "",
                         e1=e1, e2=e2, at_risk=at_risk[0, :-1].copy(), cif=cif[0].copy(),
                         surv=surv[0].copy(), curve=curve, cif30=float(cif[0, -1]))
        if bootstrap:
            b1, b2 = W.bootstrap_cells(e1, e2, n, replicates)
            bcif, bsurv, _ = W.aalen_johansen(b1, b2, np.full(replicates, float(n)))
            bcurve = W.conditional_risk_aj(bcif, bsurv)
            lo, hi = W.percentile_ci(bcurve, axis=0)
            c_lo, c_hi = W.percentile_ci(bcif[:, -1], axis=0)
            fit.boot = BootBlock(b1, b2, bcif, bsurv, bcurve, lo, hi, float(c_lo), float(c_hi))
        fits.append(fit)

    for column in VD.STRATIFIER_COLUMNS:
        total = sum(f.n for f in fits if f.stratum.column == column)
        if total != len(df):
            raise R1Error(f"the levels of {column} cover {total:,} patients, not {len(df):,}")
    if fits[0].slug != "overall" or fits[0].n != len(df):
        raise R1Error("the first stratum is not the overall row")
    return fits


def fit_strata_v1(df: pd.DataFrame, events: Events, seed: Optional[int] = VD.SEED) -> List[Any]:
    """The exact v1 path: W.fit_stratum over W.build_strata (06_window.py main() lines 2504 to 2522).

    Returns a list of W.Fit (point curve, interval, bootstrap threshold days for VD.RISK_THRESHOLDS,
    Kaplan-Meier and end-of-day sensitivity curves, conditioning-set composition by day). Works
    for any Events, and is the input W.table5_rows, W.table5b_rows and W.build_fragment expect.
    Same RNG-order convention as fit_overall_and_strata (reseeds with VD.SEED by default).
    The residual sex stratum (slug 'sex.not-recorded') is the 35 NON-BINARY patients, display
    label 'Non-binary', not unrecorded sex: take text from `fit.stratum.level`, never the slug.
    """
    if seed is not None:
        np.random.seed(seed)
    return [W.fit_stratum(stratum, mask, events.pod_event, events.pod_death, events.is_event,
                          events.is_competing, events.exit_day, events.los)
            for stratum, mask in W.build_strata(df)]


def threshold_day(curve: np.ndarray, threshold: float):
    """First day a conditional-risk curve is strictly below `threshold`; UNREACHED_DAY (31) if never.

    A 1-D curve (31,) returns an int; a 2-D curve (B, 31) returns an int array (B,). Wraps
    W.first_day_below. Never clipped to 30 and never extrapolated past it.

    This is the FIRST crossing only, even if the conditional risk later rises above `threshold`
    again (the curve is not guaranteed monotone: a day with a competing death but no event
    shrinks the conditioning set without shrinking the residual risk). v1 reports that
    separately with W.stays_below (06_window.py line 314, the "remains below" column of
    Table 5). Any analysis that reports a threshold day MUST also report "stays below through
    day 30" beside it: stays_below_from(curve, threshold, day).
    """
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 1:
        return int(W.first_day_below(arr[None, :], threshold)[0])
    return W.first_day_below(arr, threshold)


def stays_below_from(curve: np.ndarray, threshold: float, day) -> bool:
    """True if a 1-D conditional-risk curve (31,) is strictly below `threshold` on EVERY day from
    `day` through day 30; the "stays below through day 30" companion of threshold_day().

    Thin wrapper of W.stays_below (06_window.py line 314), the check behind the "remains below"
    column of Table 5. `day` is the threshold day just found (threshold_day on the same curve). A
    day of UNREACHED_DAY (31) or None ("not reached within 30 days") returns False. Only a single
    curve is accepted (use it per bootstrap replicate if a share of replicates is wanted); a 2-D
    curve raises ValueError because W.stays_below would slice rows, not days.
    """
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != DAYS:
        raise ValueError(f"stays_below_from needs a ({DAYS},) curve, got shape {arr.shape}")
    if day is None:
        return False
    return bool(W.stays_below(arr, threshold, int(day)))


def day_or_none(day) -> Optional[int]:
    """UNREACHED_DAY becomes None (JSON null, 'Not reached within 30 days'); any other day an int."""
    d = int(day)
    return None if d > VD.FOLLOWUP_DAYS else d


def index_by_slug(fits: Sequence[StratumFit]) -> Dict[str, StratumFit]:
    """{slug: fit} for a fit_overall_and_strata result. Slugs are keys, not prose: the slug
    'sex.not-recorded' is the 35 NON-BINARY patients (label 'Non-binary'); read text from
    fit.stratum.level."""
    return {f.slug: f for f in fits}


# ===========================================================================
# 5. THRESHOLD LABELS (v1's VD.threshold_label raises on 0.10 percent)
# ===========================================================================

_THRESHOLD_TEXT = {0.01: "1%", 0.005: "0.5%", 0.0025: "0.25%", 0.001: "0.10%"}


def threshold_text(threshold: float) -> str:
    """Display text for 0.01, 0.005, 0.0025 or 0.001 -> '1%', '0.5%', '0.25%', '0.10%'."""
    key = float(threshold)
    if key not in _THRESHOLD_TEXT:
        raise ValueError(f"{threshold!r} is not one of {R1_THRESHOLDS}")
    return _THRESHOLD_TEXT[key]


def threshold_slug(threshold: float) -> str:
    """Fragment-key segment: 0.005 -> 'threshold_0_5pct', 0.001 -> 'threshold_0_10pct' (v1 style)."""
    return "threshold_" + threshold_text(threshold).replace("%", "pct").replace(".", "_")


# ===========================================================================
# 6. THE SAME-DAY TREND (inputs for the year-trend analysis)
# ===========================================================================


def sameday_counts_by_year(df: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    """{year: (same-day discharges, patients with a recorded length of stay)}, 2012 to 2024.

    Same definition as v1 Table 2 primary column and 05_inout.py main() line 1639: same-day
    means LOS_POSTOP == 0 among patients whose LOS_POSTOP is recorded, grouped by OPERYR. It
    does NOT exclude patients who did not go home (that is v1's separate 'excluding' column).
    """
    los = df["LOS_POSTOP"]
    observed = los.notna()
    year = df["OPERYR"].astype("Int64")
    out: Dict[int, Tuple[int, int]] = {}
    for y in YEARS:
        in_year = observed & year.eq(y)
        out[y] = (int((in_year & los.eq(0)).sum()), int(in_year.sum()))
    return out


def build_trend_frame(df: pd.DataFrame, outcome: str = "sameday") -> pd.DataFrame:
    """The complete-case frame of the v1 adjusted trend model (05_inout.py main() lines 1799 to 1813).

    outcome "sameday" is LOS_POSTOP == 0, "disch_by_pod1" is LOS_POSTOP <= 1. Complete case means
    LOS_POSTOP recorded and AGE_PER10, MALE, BMI_PER5, ASA_GE3, LEVELS_CAT, OPERYR all present (the
    35 non-binary patients drop out through MALE): n = 111,875 for same-day. Returns a copy with
    columns OPERYR, YEAR_LINEAR (OPERYR minus 2012), TREND_OUTCOME (0/1 int) and the five
    TREND_COVARIATES. In memory only. To refit the v1 model use fit_year_logit(); for other
    designs use IO.build_design(frame, "YEAR_LINEAR", covariates), whose categorical coding is
    LEVELS_CAT dummies with '1' as reference and (if 'OPERYR' is listed) one dummy per year
    2013..2024 with 2012 as reference, plus a 'const' column.
    """
    if outcome not in ("sameday", "disch_by_pod1"):
        raise ValueError("outcome must be 'sameday' or 'disch_by_pod1'")
    los = df["LOS_POSTOP"]
    positive = los.eq(0) if outcome == "sameday" else los.le(1)
    year = df["OPERYR"].astype("Int64")
    frame = pd.DataFrame({"OPERYR": df["OPERYR"]}, index=df.index)
    frame["YEAR_LINEAR"] = year.astype("Float64") - float(YEARS[0])
    for col in TREND_COVARIATES:
        frame[col] = df[col]
    complete = los.notna()
    for col in TREND_COVARIATES:
        if col == "LEVELS_CAT":
            complete &= frame[col].astype("string").fillna("").ne("")
        else:
            complete &= frame[col].notna()
    complete &= frame["YEAR_LINEAR"].notna()
    subset = frame[complete].copy()
    subset["TREND_OUTCOME"] = positive[complete].astype(int).to_numpy()
    return subset


def fit_year_logit(frame: pd.DataFrame, covariates: Sequence[str] = TREND_COVARIATES,
                   exposure: str = "YEAR_LINEAR") -> Dict[str, Any]:
    """IO.fit_logit(frame, 'TREND_OUTCOME', exposure, covariates): the v1 logistic year-term fit.

    Returns {'n', 'events', 'beta', 'se', 'or', 'lo', 'hi', 'p', 'converged'} for the exposure term
    only (no covariate odds ratios are reported). It does not return the log-likelihood: for
    likelihood-ratio tests build the design with IO.build_design and fit sm.Logit yourself.
    """
    return IO.fit_logit(frame, "TREND_OUTCOME", exposure, list(covariates))


# ===========================================================================
# 7. OUTPUT WRITERS AND TEXT GUARDS
# ===========================================================================

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")
_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.$")
_PATIENT_ID_RE = re.compile(r"case_?uid|caseid", re.IGNORECASE)


def _check_writable(path: Path) -> Path:
    """The write guard without side effects: return the resolved path or raise R1Error.

    Only paths under RESULTS_DIR (ACDF_RESULTS_DIR, default results/revision/) are writable.
    The analysis folder is read only: RESULTS_DIR may not sit inside it (checked at import).
    """
    p = Path(path).resolve()
    if RESULTS_DIR not in p.parents:
        raise R1Error(f"refusing to write outside the results directory {RESULTS_DIR}: {p}")
    return p


def _assert_writable(path: Path) -> Path:
    """Run _check_writable and then create the parent directory. Returns the resolved path."""
    p = _check_writable(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def assert_clean_text(text: Any, where: str = "") -> str:
    """Return `text` unchanged unless it holds an em-dash (U+2014) or a snake_case token.

    Wraps VD.assert_renderable (the v1 write-time check): raises R1Error on a violation. Use it on
    every string that reaches a CSV or a figure.
    """
    try:
        return VD.assert_renderable(text, where=where)
    except ValueError as exc:
        raise R1Error(str(exc)) from exc


def _scalar(key: str, v: Any) -> Any:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if not math.isfinite(f):
            raise R1Error(f"key {key}: non-finite value; omit the key instead (suppressed or "
                          f"undefined) or pass None for 'not reached'")
        return f
    if v is None or isinstance(v, (int, str)):
        if isinstance(v, str):
            if VD.has_em_dash(v):
                raise R1Error(f"key {key}: em-dash in a string value")
            if VD.has_snake_case(v) and not _snake_case_allowed(key):
                raise R1Error(f"key {key}: string value {v!r} contains a snake_case token; "
                              f"prose in a results file must not (only keys ending in slug or "
                              f"category may carry identifiers)")
        return v
    raise R1Error(f"key {key}: value of type {type(v).__name__} is not a scalar")


def _snake_case_allowed(key: str) -> bool:
    """True for the few keys whose string value is an identifier, not prose: the LAST dotted
    segment of the key is 'slug' or 'category' or ends with '_slug' or '_category'. An example is
    a1.decision.category (value such as 'reframe_baseline'), so a last segment that is exactly
    'category' or 'slug' qualifies as well as one that ends in the suffix."""
    last = key.rsplit(".", 1)[-1]
    return last in ("slug", "category") or last.endswith(("_slug", "_category"))


def _validate_fragment(results: Mapping[str, Any], prefixes: Sequence[str]) -> Dict[str, Any]:
    prefixes = tuple(prefixes)
    if not prefixes or any(not _PREFIX_RE.match(p) for p in prefixes):
        raise R1Error(f"prefixes must look like 'a1.': {prefixes}")
    clean: Dict[str, Any] = {}
    for key, value in results.items():
        if not isinstance(key, str) or not KEY_RE.match(key):
            raise R1Error(f"key is not a flat lowercase dotted path: {key!r}")
        if not key.startswith(prefixes):
            raise R1Error(f"key {key!r} is outside the reserved prefixes {prefixes}")
        if key in clean:
            raise R1Error(f"duplicate key {key!r}")
        if isinstance(value, (list, tuple, np.ndarray)):
            clean[key] = [_scalar(key, v) for v in list(value)]
        else:
            clean[key] = _scalar(key, value)
    return clean


def write_fragment(path: Path, results: Mapping[str, Any], prefixes: Sequence[str]) -> int:
    """Write one results fragment, under RESULTS_DIR only. Returns the number of keys.

    Validates: every key matches ^[a-z0-9][a-z0-9_-]*(\\.[a-z0-9][a-z0-9_-]*)*$ and starts with one of
    `prefixes` (each like 'a1.'); values are scalars or lists of scalars (numpy converted); no NaN or
    infinity (omit the key, or pass None for 'not reached'); no em-dash in a string; and no
    snake_case token (VD.has_snake_case, any letter or digit, underscore, letter or digit) in a
    string value, because prose in a results file must not carry code names. EXCEPTION: a string
    value is allowed to be an identifier such as 'reframe_baseline' when the
    last dotted segment of its KEY is 'slug' or 'category' or ends with '_slug' or '_category'
    (for example a1.decision.category, a2.stratum_slug). Such a value is a machine label: never
    paste it into prose, take display text from stratum.level / VD.pretty(). Writes
    json.dumps(sort_keys=True, indent=2, allow_nan=False) plus a trailing newline, then reads the
    file back and raises if it does not equal what was meant to be written. Write each fragment
    once, from one process.
    """
    clean = _validate_fragment(results, prefixes)
    p = _assert_writable(Path(path))
    text = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    p.write_text(text, encoding="utf-8")
    if json.loads(p.read_text(encoding="utf-8")) != clean:
        raise R1Error(f"fragment {p.name} did not read back identically")
    return len(clean)


def write_csv_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write one aggregate CSV table, under RESULTS_DIR only, via v1's W.write_csv.

    Every header and every cell is passed through VD.assert_renderable (no em-dash, no snake_case),
    rows must match the header width, and a patient identifier column is refused. Use human-readable
    headers and VD.pretty()/VD.pretty_level() for names; write 'Suppressed' (suppressed_text) for a
    withheld estimate; give every percentage its denominator in a column.
    """
    if any(_PATIENT_ID_RE.search(str(h)) for h in header):
        raise R1Error("a patient identifier column is not allowed in an output table")
    head = [str(h) for h in header]
    for h in head:
        assert_clean_text(h, where=f"{Path(path).name} header")
    for row in rows:
        if len(row) != len(head):
            raise R1Error(f"{Path(path).name}: a row has {len(row)} cells for {len(head)} columns")
        for cell in row:
            assert_clean_text("" if cell is None else str(cell), where=f"{Path(path).name} cell")
    p = _assert_writable(Path(path))          # validate everything first so a refusal leaves no partial file
    W.write_csv(p, head, rows)


# ===========================================================================
# 8. FIGURE STYLE (consistent with v1 Figure 6)
# ===========================================================================
#
# The style follows the main-text Figure 6 of the original publication (a 2 x 2 panel figure of
# conditional-risk step curves): palette BLUE, ORANGE, BLUE_STEP (ordinal ramp), INK, INK_MUTED,
# RULE and SERIES_STYLE; spines top/right off, white faces, recessive axis edge RULE, INK labels,
# grid RULE, no legend frame, TrueType fonts; 2 x 2 panels, sharey, figsize 7.9 x 7.6; overall
# curve grey dashed (0,(5,2)) lw 1.0, threshold lines (0,(2.6,2.2)) lw 0.7, each level a step
# curve lw 1.4 in the ordinal blues with a 95 percent band alpha 0.16 and a marker every 5th day
# ms 3.6, panel letters A to D, no title.
#
# FONT SIZE. The original figure renders text at 11 to 14 pt on its 7.9 inch canvas. These figures
# are smaller on the page, so style_figure() defaults to base_pt=8.0; pass
# base_pt=V1_FIGURE_FONT_PT to match the original exactly.

BLUE = "#1c5cab"
ORANGE = "#eb6834"
BLUE_STEP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab")     # ordinal ramp, light to dark
INK = "#0b0b0b"
INK_MUTED = "#52514e"
RULE = "#c3c2b7"
SERIES_STYLE: Tuple[Tuple[str, str], ...] = (("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"))
THRESHOLD_LINESTYLE = (0, (2.6, 2.2))
OVERALL_LINESTYLE = (0, (5, 2))
FIG_SIZE_FOUR_PANEL = (7.9, 7.6)
FOUR_PANEL_ADJUST = dict(left=0.092, right=0.988, top=0.916, bottom=0.150, hspace=0.40, wspace=0.15)
V1_FIGURE_FONT_PT = 11.0


def style_figure(base_pt: float = 8.0, apply: bool = True) -> Dict[str, Any]:
    """Return (and by default apply) the matplotlib rcParams for a figure of this folder.

    Matches the original figure style (colours, spines, grid, no legend frame, TrueType fonts,
    white faces) with sans-serif text at `base_pt` (8.0 by default, panel letters and tick text a little
    larger or smaller as set by the caller). Also forces the Agg backend. No title is baked into
    an image: never call suptitle or set_title (save_png enforces it).
    """
    import matplotlib
    matplotlib.use("Agg")
    params: Dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": float(base_pt),
        "axes.labelsize": float(base_pt),
        "xtick.labelsize": float(base_pt) - 0.5,
        "ytick.labelsize": float(base_pt) - 0.5,
        "legend.fontsize": float(base_pt) - 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.6,
    }
    if apply:
        matplotlib.rcParams.update(params)
    return params


def plot_step_curve(ax, days: Sequence[float], est: Sequence[float],
                    lo: Optional[Sequence[float]] = None, hi: Optional[Sequence[float]] = None,
                    level_index: int = 0, color: Optional[str] = None,
                    label: Optional[str] = None):
    """Draw one stratum the way v1 Figure 6 does: a 'post' step line (lw 1.4), an optional step
    band (alpha 0.16) and a marker on every 5th day. Colour defaults to BLUE_STEP[level_index],
    line style and marker to SERIES_STYLE[level_index]. Returns a Line2D legend handle."""
    from matplotlib.lines import Line2D
    style, marker = SERIES_STYLE[level_index % len(SERIES_STYLE)]
    col = color or BLUE_STEP[level_index % len(BLUE_STEP)]
    if lo is not None and hi is not None:
        ax.fill_between(days, lo, hi, step="post", color=col, alpha=0.16, linewidth=0, zorder=3)
    ax.step(days, est, where="post", color=col, lw=1.4, linestyle=style, zorder=4)
    ax.plot(list(days)[::5], list(est)[::5], lw=0, marker=marker, ms=3.6, color=col,
            markeredgecolor=col, zorder=5)
    return Line2D([0], [0], color=col, lw=1.4, linestyle=style, marker=marker, ms=3.6, label=label)


def save_png(fig, path: Path, also_pdf: bool = True) -> List[Path]:
    """Save a figure as PNG at 300 dpi (bbox tight, pad 0.14 in, white), RESULTS_DIR only, after checks.

    Also writes the PDF with the same stem when also_pdf is True.
    When FIGURES is False (ACDF_FIGURES=0) nothing is written: the figure is closed and an empty
    list is returned.
    Refuses a figure that has a suptitle or any axes title (titles belong in captions), and runs
    every Text artist through assert_clean_text (no em-dash, no snake_case). The PDF carries no
    creation date and SOURCE_DATE_EPOCH is pinned, so two runs give byte-identical files.
    Closes the figure and returns the paths written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    if not FIGURES:
        plt.close(fig)
        return []
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    if fig.get_suptitle():
        raise R1Error("a figure title is baked into the image (suptitle); put it in the caption")
    for ax in fig.get_axes():
        for loc in ("left", "center", "right"):
            if ax.get_title(loc=loc):
                raise R1Error("an axes title is baked into the image; put it in the caption")
    fig.canvas.draw()
    for t in fig.findobj(Text):
        assert_clean_text(t.get_text(), where="figure text")
    p = _assert_writable(Path(path))
    kwargs = dict(bbox_inches="tight", pad_inches=0.14, facecolor="white")
    written = [p]
    fig.savefig(p, dpi=300, **kwargs)
    if also_pdf:
        pdf = _assert_writable(p.with_suffix(".pdf"))
        fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None}, **kwargs)
        written.append(pdf)
    plt.close(fig)
    return written


# ===========================================================================
# 9. SEEDS
# ===========================================================================


def seed_all() -> int:
    """np.random.seed(VD.SEED) (20260731). Returns the seed. Call it at the top of every script."""
    np.random.seed(VD.SEED)
    return VD.SEED


# ===========================================================================
# 10. SELF-TEST
# ===========================================================================

__all__ = [
    "REPO_ROOT", "ANALYSIS_DIR", "RESULTS_DIR", "ORIG_TABLES", "ORIG_OUTPUTS", "TABLES_OUT",
    "FRAGMENTS_OUT", "FIGURES_OUT", "MERGED_JSON", "FIGURES", "COHORT_PARQUET",
    "EXPECTED_COHORT_N", "REQUIRED_COLUMNS", "DAYS", "UNREACHED_DAY", "YEARS", "COVID_YEAR",
    "TREND_COVARIATES", "R1_THRESHOLDS", "suppressed_text",
    "BLUE", "ORANGE", "BLUE_STEP", "INK", "INK_MUTED", "RULE", "SERIES_STYLE", "THRESHOLD_LINESTYLE",
    "OVERALL_LINESTYLE", "FIG_SIZE_FOUR_PANEL", "FOUR_PANEL_ADJUST", "V1_FIGURE_FONT_PT",
    "VD", "W", "IO", "R1Error", "load_original_module",
    "load_cohort", "Events", "make_events", "sameday_counts_by_year", "build_trend_frame",
    "fit_year_logit", "BootBlock", "StratumFit", "fit_overall_and_strata", "fit_strata_v1",
    "threshold_day", "stays_below_from", "day_or_none", "index_by_slug", "threshold_text",
    "threshold_slug",
    "write_fragment", "write_csv_table", "assert_clean_text", "style_figure", "plot_step_curve",
    "save_png", "seed_all",
]


def _expect_raises(fn, exc=(R1Error, ValueError)) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def self_test() -> int:
    """Cohort load, make_events on the composite and on hematoma, estimator parity with
    W.fit_stratum, writer validation and the write guard. Writes nothing."""
    checks = 0

    def ok(cond: bool, msg: str) -> None:
        nonlocal checks
        if not cond:
            raise SystemExit(f"[r1_common] SELF-TEST FAILED: {msg}")
        checks += 1

    df = load_cohort()
    ok(len(df) == EXPECTED_COHORT_N, "cohort rows")

    comp = make_events(df, "WINDOW_COMP", "POD_WINDOW")
    ok(comp.n_events == 1015 and comp.n_untimed_events == 0, "composite is 1,015 timed events")
    ok(comp.n_death == 213 and comp.n_competing == 119 and comp.n_both == 94
       and comp.n_tie_same_day == 24 and comp.n_death_strictly_first == 0,
       "composite competing-event bookkeeping matches v1 (213 / 119 / 94 / 24 / 0)")

    ok(_expect_raises(lambda: make_events(df, "HEMATOMA", "HEMATOMA_POD")),
       "default untimed='raise' must raise on the untimed hematoma")
    hem = make_events(df, "HEMATOMA", "HEMATOMA_POD", untimed="drop")
    ok(hem.n_event_flagged == 541 and hem.n_events == 540 and hem.n_untimed_events == 1,
       "hematoma: 541 flagged, 540 timed events, 1 untimed dropped")
    hem30 = make_events(df, "HEMATOMA", "HEMATOMA_POD", untimed="assign", untimed_day=30)
    ok(hem30.n_events == 541 and hem30.n_untimed_events == 1, "assign keeps all 541 events")
    air = make_events(df, "AIRWAY_COMP", "POD_AIRWAY")
    ok(air.n_events == 608 and air.n_untimed_events == 0, "airway is 608 timed events")

    # Death-first rule: no real outcome has a death-first patient, and none is in both e1 and e2.
    for label, ev in (("composite", comp), ("hematoma", hem), ("airway", air)):
        ok(ev.n_death_strictly_first == 0, f"{label}: no patient dies strictly before their event")
        ok(not bool(np.any(ev.is_event & ev.is_competing)), f"{label}: is_event and is_competing are disjoint")
    # Synthetic rows: 0 death first (event day 5, death day 2), 1 same-day tie (5, 5), 2 event
    # first (3, death 8), 3 nothing, 4 untimed event (dropped). Nothing is written anywhere.
    syn = pd.DataFrame({
        "EV": [1, 1, 1, 0, 1],
        "EV_POD": [5.0, 5.0, 3.0, np.nan, np.nan],
        "MORT30": [1, 1, 1, 0, 0],
        "POD_DEATH": [2.0, 5.0, 8.0, np.nan, np.nan],
        "LOS_POSTOP": [1.0, 2.0, 3.0, 4.0, 5.0]})
    se = make_events(syn, "EV", "EV_POD", untimed="drop")
    ok(se.is_event.tolist() == [False, True, True, False, False]
       and se.is_competing.tolist() == [True, False, False, False, False],
       "synthetic death-first row is the competing event ONLY; the tie and event-first rows stay events")
    ok(not bool(np.any(se.is_event & se.is_competing)), "synthetic: exactly one of is_event / is_competing")
    ok(se.n_death_strictly_first == 1 and se.n_tie_same_day == 1 and se.n_both == 3
       and se.n_events == 2 and se.n_event_flagged == 4 and se.n_untimed_events == 1
       and se.n_competing == 1 and se.n_death == 3,
       "synthetic bookkeeping: 1 death first, 1 tie, n_both 3, 2 events of 4 flagged (1 death first, 1 untimed)")
    ok(se.exit_day[0] == 2.0 and se.exit_day[1] == 5.0 and se.exit_day[2] == 3.0
       and np.isinf(se.exit_day[3]) and np.isnan(se.pod_event[0]),
       "synthetic exit days: the death-first patient leaves on the death day with no event day")
    e1d, e2d = W.counts_by_day(se.pod_event[se.is_event], se.pod_death[se.is_competing])
    ok(e1d.sum() == 2 and e2d.sum() == 1 and e2d[2] == 1 and e1d[5] == 1 and e1d[3] == 1,
       "synthetic counts by day: the death-first patient is in e2 only")

    fits = fit_overall_and_strata(df, comp)
    v1fits = fit_strata_v1(df, comp)
    ok(len(fits) == len(v1fits) == 13, "13 strata")
    for mine, theirs in zip(fits, v1fits):
        ok(mine.slug == theirs.stratum.slug and mine.n == theirs.n and mine.events == theirs.events
           and mine.suppressed == theirs.suppressed, f"stratum identity {mine.slug}")
        if mine.suppressed:
            continue
        ok(np.array_equal(mine.curve, theirs.curve), f"point curve parity {mine.slug}")
        ok(np.array_equal(mine.boot.lo, theirs.lo) and np.array_equal(mine.boot.hi, theirs.hi),
           f"bootstrap interval parity {mine.slug}")
        for t in VD.RISK_THRESHOLDS:
            ok(np.array_equal(mine.boot_threshold_days(t), theirs.boot_days[t]),
               f"bootstrap threshold-day parity {mine.slug} {t}")
    ok(fits[0].threshold_day(0.0025) == 7 and fits[0].threshold_day(0.005) == 3,
       "overall 0.25 percent day 7 and 0.5 percent day 3")
    ok(threshold_text(0.001) == "0.10%" and threshold_slug(0.001) == "threshold_0_10pct",
       "0.10 percent threshold label")

    good = _validate_fragment({"a1.x.y_n": np.int64(3), "a1.z": [1.5, None, True]}, ("a1.",))
    ok(good == {"a1.x.y_n": 3, "a1.z": [1.5, None, True]}, "fragment validator accepts good input")
    ok(_expect_raises(lambda: _validate_fragment({"b1.x": 1}, ("a1.",))), "prefix enforced")
    ok(_expect_raises(lambda: _validate_fragment({"a1.X": 1}, ("a1.",))), "lowercase key enforced")
    ok(_expect_raises(lambda: _validate_fragment({"a1.x": float("nan")}, ("a1.",))), "NaN refused")
    ok(_expect_raises(lambda: _validate_fragment({"a1": 1}, ("a1.",))), "bare namespace key refused")
    ok(_expect_raises(lambda: assert_clean_text("a" + chr(0x2014) + "b")), "em-dash refused")
    ok(_expect_raises(lambda: assert_clean_text("risk_pct")), "snake_case refused")
    # write_fragment string values: no snake_case prose, except under *slug / *category keys.
    ok(_validate_fragment({"a1.decision.category": "reframe_baseline"}, ("a1.",))
       == {"a1.decision.category": "reframe_baseline"}, "a1.decision.category may carry reframe_baseline")
    ok(_validate_fragment({"a1.x.stratum_slug": "age_group", "a1.x.slug": "sex_x",
                           "a1.x.cut_category": ["a_b", None]}, ("a1.",))["a1.x.stratum_slug"] == "age_group",
       "keys ending in slug or category may carry identifiers (also in lists)")
    ok(_validate_fragment({"a1.x.note": "plain words, 3 days; no code names"}, ("a1.",))["a1.x.note"]
       == "plain words, 3 days; no code names", "ordinary prose string accepted")
    ok(_expect_raises(lambda: _validate_fragment({"a1.decision.note": "reframe_baseline"}, ("a1.",))),
       "snake_case string under an ordinary key refused")
    ok(_expect_raises(lambda: _validate_fragment({"a1.x.category_note": "a_b"}, ("a1.",))),
       "category in the middle of the last segment is not the exception")
    ok(_expect_raises(lambda: _validate_fragment({"a1.x.note": ["fine", "los_postop"]}, ("a1.",))),
       "snake_case inside a list of strings refused")
    ok(_expect_raises(lambda: _validate_fragment({"a1.x.note": "a" + chr(0x2014) + "b"}, ("a1.",))),
       "em-dash string still refused")

    # Write guard (never writes: refusals fire before any mkdir, acceptance uses the pure check).
    ok(_expect_raises(lambda: _assert_writable(ANALYSIS_DIR / "x.csv")), "writes under the analysis folder refused")
    ok(_expect_raises(lambda: _assert_writable(RESULTS_DIR.parent / "x.csv")), "writes outside RESULTS_DIR refused")
    probe = TABLES_OUT / "r1_selftest_probe_never_written.csv"
    ok(_check_writable(probe) == probe.resolve() and not probe.exists(),
       "a path under RESULTS_DIR is accepted (and nothing was written)")

    # threshold_day returns the first crossing only; stays_below_from is the companion check.
    bumpy = np.full(DAYS, 0.02)
    bumpy[3:] = 0.001
    bumpy[5] = 0.02                                           # dips below at day 3, rises again at day 5
    ok(threshold_day(bumpy, 0.01) == 3, "threshold_day is the first crossing (day 3) of a bumpy curve")
    ok(stays_below_from(bumpy, 0.01, 3) is False and stays_below_from(bumpy, 0.01, 6) is True
       and stays_below_from(bumpy, 0.01, 3) == W.stays_below(bumpy, 0.01, 3),
       "stays_below_from sees the return above the threshold and equals W.stays_below")
    ok(stays_below_from(bumpy, 0.01, None) is False and stays_below_from(bumpy, 0.01, UNREACHED_DAY) is False,
       "stays_below_from is False for a threshold that was never reached")
    ok(_expect_raises(lambda: stays_below_from(np.zeros((3, DAYS)), 0.01, 0)),
       "stays_below_from refuses a 2-D curve")

    print(f"[r1_common] self-test: {checks} checks PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Shared loader for the additional analyses in analysis/revision/.")
    ap.add_argument("--self-test", action="store_true", help="run the self-test and exit (writes nothing)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
