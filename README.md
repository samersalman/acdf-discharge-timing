# ACDF discharge timing: analysis code

Cohort construction and statistical analysis for a study of **discharge timing after
anterior cervical discectomy and fusion (ACDF)** in the ACS-NSQIP Adult Participant Use
Data File, **2012 to 2024**.

The study asks how much of the 30-day complication burden falls *after* the patient's own
discharge, and when the residual risk becomes small: for each risk stratum it estimates the
residual-risk threshold day, the first postoperative day on which the conditional risk of a
subsequent event, given that the patient is event free and alive at the start of the day, falls
below a pre-specified threshold. Death is handled as a competing event.

**This repository contains no patient-level data.** See [Data availability](#data-availability).

---

## Data availability

The ACS-NSQIP Participant Use Data File **is not included here and cannot be**. It is
governed by a Data Use Agreement with the American College of Surgeons that prohibits
redistribution. The MIT license (`LICENSE`) covers the source code in this repository only; it
does not and cannot grant any right to the data. To run this pipeline you must obtain the PUF
yourself:

> <https://www.facs.org/quality-programs/data-and-registries/acs-nsqip/>

The American College of Surgeons NSQIP and the hospitals participating in it are the source
of the data used; they have not verified and are not responsible for the statistical
validity of the data analysis or the conclusions drawn by the authors.

Step `01` expects a local DuckDB warehouse built from the PUF, exposing an `export.py`
that accepts a variable list and a year range. Point the pipeline at it with:

```bash
export NSQIP_WAREHOUSE=/path/to/your/nsqip-db     # defaults to ~/nsqip-db
```

---

## Layout

```
analysis/           the pipeline, in execution order
  01_export_cohort.py     pull the identified cohort from the warehouse (read-only)
  02_build_cohort.py      recode and freeze the analytic cohort; the correctness gate
  03_timing.py            Aalen-Johansen cumulative incidence, Gray's test, Tables 3A/S1/S3
  04_postdischarge.py     post-discharge event fractions and the denominator ladder
  05_inout.py             length of stay, discharge-timing trend, the INOUT billing critique
  06_window.py            conditional risk by stratum, the residual-risk threshold day
  cpt_acdf.py             CPT and ICD taxonomy; the ONLY place a code literal may appear
  variable_dict.py        labels, level maps, formatters, and the project seed
  vars/                   the 136-column NSQIP export variable list
  revision/               the additional analyses of the Supplement (see below)
    r1_common.py            shared loader and helpers; reads the original scripts, never edits them
    a1_baseline_vs_residual.py ... a9_convention_sensitivity.py   one script per analysis
    merge_results_r1.py     merges the per-analysis fragments into results_r1.json

results/            the authors' published output, aggregate only, no patient rows
  results.json            4,532 traceable result keys from the published run
  tables-csv/             the 24 published tables and cumulative-incidence curves
  revision/               output of the additional analyses, aggregate only
    results_r1.json         merged result keys of the additional analyses
    tables-csv/             the 13 Supplement tables they produce
```

## Install and run

```bash
pip install -r requirements.txt
export NSQIP_WAREHOUSE=/path/to/your/nsqip-db
cd analysis

python3 01_export_cohort.py     # warehouse -> exports/     (needs the PUF)
python3 02_build_cohort.py      # exports/  -> derived/     (freezes the cohort)

# 03 to 06 are independent of one another and may run in any order.
# All four read only the frozen derived/cohort_analytic.parquet.
python3 03_timing.py
python3 04_postdischarge.py
python3 05_inout.py
python3 06_window.py
```

Each step writes a results fragment to `analysis/outputs/results_NN.json` and its tables to
`analysis/tables-csv/`. Those directories are gitignored: they are *your* run, not the
published record.

**To check your reproduction**, compare what you produced against the reference copy:

```bash
diff <(sort analysis/tables-csv/table3a_complication_timing.csv) \
     <(sort ../results/tables-csv/table3a_complication_timing.csv)
```

`cpt_acdf.py` carries a self-test and needs no data:

```bash
python3 cpt_acdf.py     # exits 0 on success
```

## Additional analyses (Supplement)

`analysis/revision/` holds the code for the additional analyses reported in the Supplement. They
reuse the original estimator without changing it: `r1_common.py` imports `06_window.py`,
`05_inout.py` and `variable_dict.py` from the analysis folder by path, reads the frozen
`derived/cohort_analytic.parquet`, and keeps the original seed (20260731), the original order of
bootstrap draws and the original suppression rule (fewer than 10 events), so the composite
reproduces the published Table 5 cell for cell. (In these scripts "v1" means the original
pipeline of steps 01 to 06, release v1.0.0.)

| Script | Analysis | Output (under `results/revision/`) |
|---|---|---|
| `a1_baseline_vs_residual.py` | A1 baseline versus residual risk; A6 paired bootstrap on the spread of threshold days | `tables-csv/tableS5_*.csv`; Figure S1 |
| `a2_component_residual.py` | A2 residual risk of hematoma and of airway morbidity separately; the one untimed hematoma | `tables-csv/tableS6_*.csv`; Figure S2 |
| `a3_horizon_artifact.py` | A3 discrete-time hazard by band and a horizon-matched 16-day window | `tables-csv/tableS7_*.csv`; Figure S3 |
| `a4_year_trend.py` | A4 year-trend diagnostics for the same-day discharge trend | `tables-csv/tableS8_*.csv`; Figure S4 |
| `a5_direct_standardization.py` | A5 adjusted strata by direct standardization | `tables-csv/tableS9_adjusted_strata.csv` |
| `a7_fixed_day_table.py` | A7 residual risk at fixed days (reformats Table 5B, no recomputation) | `tables-csv/tableS10_fixed_day_risk.csv` |
| `a8_selection_riskset.py` | A8 who is still admitted: descriptive selection in the conditioning set | `tables-csv/tableS11_selection_still_admitted.csv` |
| `a9_convention_sensitivity.py` | A9 sensitivity to the same-day tie rule and to the conditioning set | `tables-csv/tableS4F_*.csv` |
| `merge_results_r1.py` | merges the per-analysis fragments | `results_r1.json` |

**Before you run them**, run steps `01` and `02` (they build `derived/cohort_analytic.parquet`) and
then `03_timing.py` and `06_window.py`. The additional scripts read what those two steps wrote,
`outputs/results_03.json`, `outputs/results_06.json` and `tables-csv/table5*.csv`, and cross-check
their own recomputation of the published Table 5 and Table 5B against it before reporting anything.

```bash
export PYTHONDONTWRITEBYTECODE=1
cd analysis/revision                 # from the repository root, after steps 01, 02, 03 and 06

python3 r1_common.py --self-test     # optional; checks the loader and the estimator parity, writes nothing

# a1 to a5, a7 and a8 are independent of one another; run a8 before a9, which reads its fragment.
python3 a1_baseline_vs_residual.py
python3 a2_component_residual.py
python3 a3_horizon_artifact.py
python3 a4_year_trend.py
python3 a5_direct_standardization.py
python3 a7_fixed_day_table.py
python3 a8_selection_riskset.py
python3 a9_convention_sensitivity.py
python3 merge_results_r1.py          # after all of the above
```

Three optional environment variables control where things are read and written:

| Variable | Meaning | Default |
|---|---|---|
| `ACDF_ANALYSIS_DIR` | folder holding `06_window.py`, `05_inout.py`, `variable_dict.py` and `derived/cohort_analytic.parquet`; only ever read | this repository's `analysis/` |
| `ACDF_RESULTS_DIR` | where every output is written; the scripts refuse to write anywhere else, and refuse a folder inside the analysis folder | `results/revision/` |
| `ACDF_FIGURES` | set to `0` to skip the PNG and PDF figures; every number and table is unchanged | figures on |

Outputs land in `results/revision/`: `tables-csv/` (the Supplement tables), `fragments/` (one
`results_aN.json` per script, the input of the merge) and `figures/`. Only `results_r1.json` and
`tables-csv/` are the published record; `fragments/` and `figures/` are products of your own run
and are gitignored, like `analysis/outputs/`. Unlike `results/results.json`,
`results/revision/results_r1.json` **can** be regenerated from this repository, because the merge
step is included.

A run with the default settings rewrites `results/revision/tables-csv/` and `results_r1.json` in
place, so an exact reproduction leaves `git status` clean. To keep your own run apart from the
reference copies and compare the two, point `ACDF_RESULTS_DIR` somewhere else:

```bash
export ACDF_RESULTS_DIR=$HOME/acdf-my-run     # then run the scripts above
diff <(sort ../../results/revision/tables-csv/tableS10_fixed_day_risk.csv) \
     <(sort "$ACDF_RESULTS_DIR"/tables-csv/tableS10_fixed_day_risk.csv)
```

These analyses need `statsmodels`, `scipy` and `patsy` (all scripts) and `matplotlib` (the figures
only); see `requirements.txt`. Everything is deterministic: two runs of the same code on the same
cohort and the same software versions give byte-identical CSV, JSON, PNG and PDF files.

**Terminology.** The paper's term is *residual-risk threshold day*. The original scripts of steps
01 to 06 predate that wording and still say "safe discharge" in some places: the console messages
and the notes file written by `06_window.py`, the file name `table5_safe_discharge_day.csv`, some
row labels of `s4_sensitivity_06.csv`, and some result keys and descriptions in `results.json`
(for example `window.sensitivity.km_safe_days_moved`). The column headers of the published
tables already use the current term. Those names are kept as they are so that the scripts and the
published record stay in step. In the paper and in this section, the quantity is a descriptive
threshold crossing in a cohort in which almost every patient still at risk had already gone home,
not a day on which discharge is established to be safe.

---

## The cohort ladder

Your rebuild should land on these counts at every rung. This is the primary specification
(`AGE` cast as in `02_build_cohort.py`; the reconciliation variant is one row-cast apart and
is reported alongside it in the paper).

| Step | n |
|---|---:|
| Principal CPT in `cpt_acdf.PRINCIPAL_CPTS`, age >= 18, 2012 to 2024 | 124,344 |
| and elective | 121,210 |
| minus exclusion CPTs in any of the ten concurrent procedure slots | 113,469 |
| minus ASA 5, ASA missing, or preoperative sepsis | **112,642** |

If your first rung is not 124,344, the divergence is in the export, not in the analysis.

## The eight timed outcomes

Hematoma requiring reoperation; any unplanned reoperation; unplanned reintubation;
ventilator dependence beyond 48 hours; pneumonia; unplanned readmission; death within 30
days; and composite airway morbidity. Authoritative definitions, code sets and the years
each applies are in `results/tables-csv/s1_field_definitions.csv`.

Cumulative incidence is Aalen-Johansen with **death as a competing event**, not one minus
Kaplan-Meier. The size of that correction is measured rather than assumed small and is
published in `s3_km_vs_aalen_johansen.csv`.

---

## What is not in this repository, and why

This repository is scoped to the **analysis**. Deliberately withheld:

| Withheld | Reason |
|---|---|
| The ACS-NSQIP PUF, the export bundle, and both derived cohort parquets | Patient-level data. Data Use Agreement. |
| Figure generation for the main-text figures (`07_figures.py`) | Presentation, not analysis. The code for Supplement Figures S1 to S4 is part of the additional analyses (`analysis/revision/`). |
| Manuscript generation (`08_generate_manuscript.py`) and the .docx toolchain | Manuscript production, not analysis. |
| The pipeline runner (`run_all.py`) | Not analysis. Its execution order is documented above instead. |
| The number-traceability gate (`verify_numbers.py`) | Not analysis. |

Two consequences worth stating plainly:

1. **`results/results.json` cannot be regenerated from this repository.** The step that
   merged the per-step fragments into it lived in the withheld runner. The file here
   is a static artefact of the authors' run. The individual fragments that feed it,
   `outputs/results_NN.json`, *are* reproducible. (The corresponding file of the additional
   analyses, `results/revision/results_r1.json`, is regenerated by `merge_results_r1.py`.)
2. **`05_inout.py` skips one self-check.** Its `verify_fragment()` re-read the fragment
   through the runner's namespace validator. With the runner withheld, that
   import is guarded and the check is skipped with a printed notice. The fragment is written
   either way; only the namespace assertion is unavailable.

## Reproducibility caveat

The published run used **CPython 3.14**, recorded in the pipeline run log. Its package
versions were not captured at run time and that environment has since been cleared, so the
exact versions behind the published numbers are not recoverable. `requirements.txt` pins a
verified-working set under CPython 3.12.10. Competing-risks estimates can shift slightly
between `lifelines` releases, so please report the versions you actually used.

The additional analyses in `analysis/revision/` were run repeatedly under CPython 3.14.0 with
pandas 2.3.3, numpy 2.4.2, scipy 1.17.0, statsmodels 0.14.6, patsy 1.0.2, lifelines 0.30.3,
pyarrow 23.0.0 and matplotlib 3.10.8, and every run gave byte-identical outputs. They were not
rerun under the CPython 3.12.10 set pinned in `requirements.txt`.

## Citing

See `CITATION.cff`. Code is MIT licensed (`LICENSE`); see Data availability for what the
license does not cover.
