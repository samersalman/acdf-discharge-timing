#!/usr/bin/env python3
"""a7_fixed_day_table.py - A7, fixed-day presentation of residual risk.

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a7_fixed_day_table.py

Reformats tables-csv/table5b_conditional_risk.csv of the analysis folder (written by step 06; NOT
recomputed, no RNG, no cohort) into Table S10: one row per stratum, residual risk with its 95%
interval and the number of patients event free and alive at the start of the day, at days 1, 2, 3,
5, 7, 10, 14 and 21. Suppressed strata stay suppressed with their counts. Every printed Table 5 cell that falls on one of these
days is cross-checked against the reformatted value.

Writes (only these, under the results directory): tables-csv/tableS10_fixed_day_risk.csv and
fragments/results_a7.json (namespace a7.).
"""

import sys; sys.dont_write_bytecode = True
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv

import r1_common as R1

VD, W = R1.VD, R1.W

DAYS_SHOWN = (1, 2, 3, 5, 7, 10, 14, 21)
EN = chr(0x2013)
SRC_5B = R1.ORIG_TABLES / "table5b_conditional_risk.csv"
SRC_5 = R1.ORIG_TABLES / "table5_safe_discharge_day.csv"

COL_DAY = "Postoperative day"
COL_RISK = "Conditional risk of a subsequent composite event (%)"
COL_LO = "95% confidence interval, lower (%)"
COL_HI = "95% confidence interval, upper (%)"
COL_AT = "Patients event free and alive at the start of the day"
COL_CIF = "Cumulative incidence of the composite to the start of the day (%)"


def risk_text(risk, lo, hi):
    return f"{float(risk):.3f} ({float(lo):.3f}{EN}{float(hi):.3f})"


def expected_strata():
    """[(slug, stratifier display, stratum display)] in display order, built from the v1 label layer."""
    out = [("overall", "Overall", "All patients")]
    for spec in VD.STRATIFIERS:
        col = spec.column
        for raw in spec.levels:
            out.append((f"{W.STRATIFIER_SLUG[col]}.{W.level_slug(col, raw)}",
                        W.STRATIFIER_DISPLAY[col], VD.pretty_level(col, raw)))
        if col == "MALE":                           # the residual sex level: 35 non-binary patients
            out.append((f"{W.STRATIFIER_SLUG[col]}.{W.RESIDUAL_SLUG}", W.STRATIFIER_DISPLAY[col], "Non-binary"))
    return out


def main():
    with open(SRC_5B, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = expected_strata()
    by_order = {}
    for r in rows:
        by_order.setdefault(int(r["Display order"]), []).append(r)
    if sorted(by_order) != list(range(len(expected))):
        raise R1.R1Error(f"Table 5B carries display orders {sorted(by_order)}, expected 0 to {len(expected) - 1}")

    strata = []
    for order, (slug, stratifier, level) in enumerate(expected):
        block = by_order[order]
        if [int(r[COL_DAY]) for r in block] != list(range(W.DAYS)):
            raise R1.R1Error(f"{slug}: Table 5B does not carry days 0 to 30 in order")
        if block[0]["Stratifier"] != stratifier or block[0]["Stratum"] != level:
            raise R1.R1Error(f"display order {order}: Table 5B says {block[0]['Stratifier']} / {block[0]['Stratum']}, "
                             f"expected {stratifier} / {level}")
        strata.append((slug, stratifier, level, block))

    header = ["Stratifier", "Stratum", "Patients", "Composite events", "denominator",
              "30-day cumulative incidence, % (95% CI)"]
    for d in DAYS_SHOWN:
        header.append(f"Day {d}: residual risk, % (95% CI)")
        header.append(f"Day {d}: patients event free and alive (n)")

    res = {}
    table = []
    cross = 0
    with open(SRC_5, encoding="utf-8", newline="") as fh:
        t5 = list(csv.DictReader(fh))
    for slug, stratifier, level, block in strata:
        n, ev = int(block[0]["Patients"]), int(block[0]["Composite events"])
        suppressed = block[0]["Suppressed"] == "Yes"
        head = [stratifier, level, VD.fmt_count(n), VD.fmt_count(ev), VD.fmt_count(int(block[0]["denominator"]))]
        p = f"a7.{slug}."
        res[p + "patients_n"] = n
        res[p + "events_n"] = ev
        res[p + "suppressed"] = bool(suppressed)
        if suppressed:
            if ev >= VD.MIN_STRATUM_EVENTS:
                raise R1.R1Error(f"{slug} is flagged suppressed with {ev} events")
            table.append(head + [W.SUPPRESSED_TEXT] * (len(header) - len(head)))
            continue
        d0 = block[0]
        cells = [risk_text(d0[COL_RISK], d0[COL_LO], d0[COL_HI])]
        res[p + "cif30_pct"] = float(d0[COL_RISK])
        res[p + "cif30_lo_pct"] = float(d0[COL_LO])
        res[p + "cif30_hi_pct"] = float(d0[COL_HI])
        for d in DAYS_SHOWN:
            r = block[d]
            if int(r[COL_DAY]) != d:
                raise R1.R1Error("Table 5B day order broke")
            cells.append(risk_text(r[COL_RISK], r[COL_LO], r[COL_HI]))
            cells.append(VD.fmt_count(int(r[COL_AT])))
            q = f"{p}day_{d}."
            res[q + "risk_pct"] = float(r[COL_RISK])
            res[q + "ci_lo_pct"] = float(r[COL_LO])
            res[q + "ci_hi_pct"] = float(r[COL_HI])
            res[q + "at_risk_n"] = int(r[COL_AT])
        table.append(head + cells)

        # cross-check against the printed Table 5 for every threshold row whose day is one of ours,
        # and the 30-day incidence against the day-0 row
        for row in t5:
            if row["Stratifier"] != stratifier or row["Stratum"] != level:
                continue
            if row["Cumulative incidence of the composite to day 30 (%)"] != f"{float(d0[COL_RISK]):.3f}%":
                raise R1.R1Error(f"{slug}: Table 5 30-day incidence differs from the Table 5B day-0 row")
            day_text = row["Residual-risk threshold day"]
            if not day_text.isdigit():
                continue
            day = int(day_text)
            r = block[day]
            printed = row["Residual risk at that day (%)"]
            ci = row["Residual risk, 95% confidence interval (%)"]
            at = row["Patients event free and alive at the start of that day"]
            mine = f"{float(r[COL_RISK]):.3f}%"
            mine_ci = f"{float(r[COL_LO]):.3f}{EN}{float(r[COL_HI]):.3f}"
            if printed != mine or ci != mine_ci or at != VD.fmt_count(int(r[COL_AT])):
                raise R1.R1Error(f"{slug} day {day}: Table 5 prints {printed} {ci} {at}, Table 5B gives {mine} {mine_ci}")
            cross += 1

    R1.write_csv_table(R1.TABLES_OUT / "tableS10_fixed_day_risk.csv", header, table)

    # ---- headline numbers ------------------------------------------------------------------------------
    ov = next(b for s, _a, _b, b in strata if s == "overall")
    cif = float(ov[0][COL_RISK])                                  # percent, 0.901085
    res["a7.overall.cif30_pct"] = cif
    res["a7.overall.cif30_rounded_3dp_pct"] = round(cif, 3)
    res["a7.overall.cif30_rounded_2dp_pct"] = round(cif, 2)
    res["a7.check.cohort_incidence_matches_0_90_pct"] = bool(round(cif, 2) == 0.90)
    res["a7.check.cohort_incidence_matches_0_901_pct"] = bool(round(cif, 3) == 0.901)
    for t in ABS_THR():
        q = f"a7.thresholds.{R1.threshold_slug(t)}."
        res[q + "threshold_pct"] = 100.0 * t
        res[q + "of_baseline_prop"] = 100.0 * t / cif
        res[q + "of_baseline_pct"] = 100.0 * (100.0 * t / cif)
    res["a7.days_shown"] = list(DAYS_SHOWN)
    res["a7.check.table5_cells_crosschecked_n"] = cross
    res["a7.strata_n"] = len(strata)
    n_keys = R1.write_fragment(R1.FRAGMENTS_OUT / "results_a7.json", res, ("a7.",))

    print(f"[a7] wrote {n_keys} keys and {len(table)} table rows; {cross} Table 5 cells cross-checked against Table 5B")
    print(f"[a7] overall 30-day cumulative incidence {cif:.6f}% (rounds to {round(cif, 2):.2f} at 2 decimals, {round(cif, 3):.3f} at 3)")
    for t in ABS_THR():
        q = f"a7.thresholds.{R1.threshold_slug(t)}."
        print(f"[a7] threshold {R1.threshold_text(t)} = {res[q + 'of_baseline_prop']:.4f} of the cohort baseline")
    for d in DAYS_SHOWN:
        q = f"a7.overall.day_{d}."
        print(f"[a7] overall day {d}: {res[q + 'risk_pct']:.3f}% ({res[q + 'ci_lo_pct']:.3f} to {res[q + 'ci_hi_pct']:.3f}), "
              f"{res[q + 'at_risk_n']:,} event free and alive")
    return 0


def ABS_THR():
    return tuple(VD.RISK_THRESHOLDS)


if __name__ == "__main__":
    raise SystemExit(main())
