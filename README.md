# ACDF discharge timing: analysis code

Cohort construction and statistical analysis for a study of **discharge timing after
anterior cervical discectomy and fusion (ACDF)** in the ACS-NSQIP Adult Participant Use
Data File, **2012 to 2024**.

The study asks when it is safe to send a patient home: it measures how much of the 30-day
complication burden falls *after* the patient's own discharge, and estimates a
risk-stratified safe discharge day from conditional risk given event-free survival, with
death handled as a competing event.

**This repository contains no patient-level data.** See [Data availability](#data-availability).

---

## Data availability

The ACS-NSQIP Participant Use Data File **is not included here and cannot be**. It is
governed by a Data Use Agreement with the American College of Surgeons that prohibits
redistribution. To run this pipeline you must obtain the PUF yourself:

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
  06_window.py            conditional risk by stratum, the safe discharge day
  cpt_acdf.py             CPT and ICD taxonomy; the ONLY place a code literal may appear
  variable_dict.py        labels, level maps, formatters, and the project seed
  vars/                   the 136-column NSQIP export variable list

results/            the authors' published output, aggregate only, no patient rows
  results.json            4,532 traceable result keys from the published run
  tables-csv/             the 24 published tables and cumulative-incidence curves
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
| Figure generation (`07_figures.py`) | Presentation, not analysis. |
| Manuscript generation (`08_generate_manuscript.py`) and the .docx toolchain | Manuscript production, not analysis. |
| The pipeline orchestrator (`run_all.py`) | Not analysis. Its execution order is documented above instead. |
| The number-traceability gate (`verify_numbers.py`) | Not analysis. |

Two consequences worth stating plainly:

1. **`results/results.json` cannot be regenerated from this repository.** The step that
   merged the per-step fragments into it lived in the withheld orchestrator. The file here
   is a static artefact of the authors' run. The individual fragments that feed it,
   `outputs/results_NN.json`, *are* reproducible.
2. **`05_inout.py` skips one self-check.** Its `verify_fragment()` re-read the fragment
   through the orchestrator's namespace validator. With the orchestrator withheld, that
   import is guarded and the check is skipped with a printed notice. The fragment is written
   either way; only the namespace assertion is unavailable.

## Reproducibility caveat

The published run used **CPython 3.14**, recorded in the pipeline run log. Its package
versions were not captured at run time and that environment has since been cleared, so the
exact versions behind the published numbers are not recoverable. `requirements.txt` pins a
verified-working set under CPython 3.12.10. Competing-risks estimates can shift slightly
between `lifelines` releases, so please report the versions you actually used.

## Citing

See `CITATION.cff`. Code is MIT licensed (`LICENSE`); the license covers this source code
only and grants no rights to the NSQIP data.
