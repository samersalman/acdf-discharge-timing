#!/usr/bin/env python3
"""a4_year_trend.py - A4 year-trend diagnostics for the same-day discharge trend.

Question. v1 reports the adjusted same-day discharge odds ratio per year (1.1497, 95% CI 1.1434
to 1.1560) from a logistic model with operative year as ONE linear term. Table 2 rises to 2022
and then falls, so a single linear term may fit badly. This script
refits that model and four alternatives that differ ONLY in the year term, on the same
complete-case rows and the same covariates as v1 05_inout.py (build_design, main() lines 1799 to
1813): linear year, categorical year (reference 2012), quadratic year, natural cubic spline of
year (4 df, and 5 df as a sensitivity).

Scope. Only YEAR terms are reported here: the linear OR, the
year-specific ORs of the categorical model, the quadratic and spline year terms and contrasts
among year terms. No covariate coefficient and no covariate odds ratio is written anywhere, no
discrimination statistic is computed, and no model is fitted for any outcome other than same-day
discharge.

Outputs (all aggregate, under the results directory): fragments/results_a4.json (namespace a4.),
tables-csv/tableS8_year_trend_models.csv (S8A), tables-csv/tableS8_year_by_year.csv (S8B),
figures/figS4_year_trend_diagnostics.png and .pdf (skipped when ACDF_FIGURES=0).

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a4_year_trend.py
"""

import sys; sys.dont_write_bytecode = True
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import math

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

import r1_common as R1

VD, W, IO = R1.VD, R1.W, R1.IO
R1.seed_all()

PREFIX = ("a4.",)
Z95 = float(stats.norm.ppf(0.975))
CENTRE_YEAR = 2018          # quadratic year is centred at the middle of 2012 to 2024
YEAR0 = R1.YEARS[0]         # 2012, the reference year and the zero of YEAR_LINEAR
V1_OR = (1.1497, 1.1434, 1.1560)    # the published value (results_05.json) at 4 decimals, checked below

R = {}                      # the JSON fragment


def put(key, value):
    if key in R:
        raise R1.R1Error(f"duplicate key {key}")
    R[key] = value


# ---------------------------------------------------------------------------
# 1. The data: the v1 complete-case frame (n = 111,875)
# ---------------------------------------------------------------------------

df = R1.load_cohort()
sub = R1.build_trend_frame(df, "sameday")
n_all = len(df)
n = len(sub)
y = sub["TREND_OUTCOME"].to_numpy(dtype=float)
year = sub["OPERYR"].astype(int).to_numpy()
if n != 111_875:
    raise R1.R1Error(f"the trend frame has {n:,} rows, expected 111,875")

# Who falls out of the frame, and why (verified from the data).
los_missing = df["LOS_POSTOP"].isna()
male_missing = df["MALE"].isna()
bmi_missing = df["BMI_PER5"].isna()
other_missing = (df["AGE_PER10"].isna() | df["ASA_GE3"].isna() | df["OPERYR"].isna()
                 | df["LEVELS_CAT"].astype("string").fillna("").eq(""))
dropped = los_missing | male_missing | bmi_missing | other_missing
if int(dropped.sum()) != n_all - n:
    raise R1.R1Error("the dropout reasons do not add up to the frame difference")
reasons_per_patient = (los_missing.astype(int) + male_missing.astype(int) + bmi_missing.astype(int)
                       + other_missing.astype(int))
put("a4.cohort.n", n)
put("a4.cohort.events_n", int(y.sum()))
put("a4.cohort.rate_pct", 100.0 * float(y.mean()))
put("a4.cohort.all_patients_n", n_all)
put("a4.cohort.excluded_n", int(dropped.sum()))
put("a4.cohort.excluded_no_recorded_stay_n", int(los_missing.sum()))
put("a4.cohort.excluded_non_binary_sex_n", int(male_missing.sum()))
put("a4.cohort.excluded_missing_bmi_n", int(bmi_missing.sum()))
put("a4.cohort.excluded_other_missing_n", int(other_missing.sum()))
put("a4.cohort.excluded_more_than_one_reason_n", int((reasons_per_patient >= 2).sum()))

t2 = R1.sameday_counts_by_year(df)          # Table 2 basis: every patient with a recorded stay
t2_events = sum(v[0] for v in t2.values())
t2_n = sum(v[1] for v in t2.values())
put("a4.table2.n", int(t2_n))
put("a4.table2.events_n", int(t2_events))
put("a4.table2.rate_pct", 100.0 * t2_events / t2_n)

# ---------------------------------------------------------------------------
# 2. The five designs (same covariates, only the year term differs)
# ---------------------------------------------------------------------------

linear_design = IO.build_design(sub, "YEAR_LINEAR", list(R1.TREND_COVARIATES))     # v1's exact design
covariates = linear_design.drop(columns="EXPOSURE")       # const + the five v1 covariates (LEVELS as dummies)
if list(linear_design.columns) != ["const", "EXPOSURE", "AGE_PER10", "MALE", "BMI_PER5", "ASA_GE3",
                                   "LEVELS.2", "LEVELS.>=3"]:
    raise R1.R1Error(f"unexpected v1 design columns: {list(linear_design.columns)}")
YEAR_DUMMIES = [f"YEAR.{yr}" for yr in R1.YEARS[1:]]


def spline_basis(x, df_spline):
    """patsy natural cubic regression spline with a centring constraint, no intercept column.

    cr(x, df=k, constraints='center') returns k columns that do NOT span the constant, so
    with the model intercept the year effect has exactly k degrees of freedom and the linear
    year term lies inside the span (checked below). Returns (matrix, knots).
    """
    state = patsy.dmatrix(f"0 + cr(x, df={df_spline}, constraints='center')", {"x": x},
                          return_type="dataframe")
    knots_transform = patsy.mgcv_cubic_splines.CR()
    knots_transform.memorize_chunk(x, df=df_spline, constraints="center")
    knots_transform.memorize_finish()
    knots = [float(k) for k in knots_transform._all_knots]
    check = patsy.build_design_matrices([state.design_info], {"x": x}, return_type="dataframe")[0]
    if not np.allclose(state.to_numpy(), check.to_numpy(), atol=1e-12):
        raise R1.R1Error("the patsy spline basis is not reproducible")
    return state.to_numpy(), knots


MODELS = ["linear", "categorical", "quadratic", "spline4", "spline5"]
LABELS = {
    "linear": "Linear year",
    "categorical": "Categorical year (13 levels, reference 2012)",
    "quadratic": "Quadratic year (year centred at 2018 and its square)",
    "spline4": "Natural cubic spline of year, 4 degrees of freedom",
    "spline5": "Natural cubic spline of year, 5 degrees of freedom",
}
YEAR_PARAMETERS = {"linear": 1, "categorical": 12, "quadratic": 2, "spline4": 4, "spline5": 5}

x_year = year.astype(float)
basis4, knots4 = spline_basis(x_year, 4)
basis5, knots5 = spline_basis(x_year, 5)


def year_terms(kind, years_vec):
    """The year columns of `kind` for an integer array of years (same row order as `sub`)."""
    years_vec = np.asarray(years_vec)
    if kind == "linear":
        return pd.DataFrame({"EXPOSURE": years_vec.astype(float) - YEAR0}, index=sub.index)
    if kind == "categorical":
        return pd.DataFrame({name: (years_vec == yr).astype(float)
                             for name, yr in zip(YEAR_DUMMIES, R1.YEARS[1:])}, index=sub.index)
    if kind == "quadratic":
        centred = years_vec.astype(float) - CENTRE_YEAR
        return pd.DataFrame({"YEAR.centred": centred, "YEAR.centred squared": centred ** 2},
                            index=sub.index)
    raise ValueError(kind)


def full_design(kind, years_vec):
    """Covariates plus the year terms of `kind`. Spline designs use the fitted basis (no refit)."""
    if kind == "linear":
        # the exact v1 design (const, EXPOSURE, covariates), built by v1's own function
        if np.array_equal(np.asarray(years_vec), year):
            return linear_design
        out = linear_design.copy()
        out["EXPOSURE"] = np.asarray(years_vec, dtype=float) - YEAR0
        return out
    if kind in ("spline4", "spline5"):
        if not np.array_equal(np.asarray(years_vec), year):
            raise R1.R1Error("spline designs are only built on the observed years")
        b = basis4 if kind == "spline4" else basis5
        cols = pd.DataFrame(b, index=sub.index, columns=[f"SPLINE.{i + 1}" for i in range(b.shape[1])])
        return pd.concat([covariates, cols], axis=1)
    return pd.concat([covariates, year_terms(kind, years_vec)], axis=1)


def fit_model(kind):
    design = full_design(kind, year)
    rank = int(np.linalg.matrix_rank(design.to_numpy()))
    if rank != design.shape[1]:
        raise R1.R1Error(f"{kind}: design is rank deficient ({rank} of {design.shape[1]})")
    res = sm.Logit(y, design).fit(disp=0, maxiter=200)
    if not res.mle_retvals.get("converged", True):
        raise R1.R1Error(f"{kind}: the logistic fit did not converge; an unconverged fit is never reported")
    k = design.shape[1]
    aic = -2.0 * float(res.llf) + 2.0 * k
    bic = -2.0 * float(res.llf) + k * math.log(n)
    if abs(aic - float(res.aic)) > 1e-6 or abs(bic - float(res.bic)) > 1e-6:
        raise R1.R1Error(f"{kind}: AIC or BIC from the log-likelihood differs from statsmodels")
    return res, design, k, aic, bic


fits = {m: fit_model(m) for m in MODELS}

# ---------------------------------------------------------------------------
# 3. Reproduce the v1 linear-year odds ratio. A miss stops the run.
# ---------------------------------------------------------------------------

lin_res = fits["linear"][0]
v1 = R1.fit_year_logit(sub)
beta = float(lin_res.params["EXPOSURE"])
se = float(lin_res.bse["EXPOSURE"])
ci = lin_res.conf_int().loc["EXPOSURE"]
lin_or, lin_lo, lin_hi = math.exp(beta), math.exp(float(ci[0])), math.exp(float(ci[1]))
if abs(lin_or - v1["or"]) > 1e-10 or abs(lin_lo - v1["lo"]) > 1e-10 or abs(lin_hi - v1["hi"]) > 1e-10:
    raise R1.R1Error("the statsmodels refit differs from the v1 fit_logit result; STOP and report")
if (round(lin_or, 4), round(lin_lo, 4), round(lin_hi, 4)) != V1_OR:
    raise R1.R1Error(f"the linear OR {lin_or:.6f} ({lin_lo:.6f} to {lin_hi:.6f}) does not read "
                     f"{V1_OR} at 4 decimals; STOP and report")
put("a4.linear.or_per_year", lin_or)
put("a4.linear.or_lo", lin_lo)
put("a4.linear.or_hi", lin_hi)
put("a4.linear.beta_per_year", beta)
put("a4.linear.se_per_year", se)
put("a4.linear.p", float(lin_res.pvalues["EXPOSURE"]))
put("a4.linear.reproduces_v1", True)

# ---------------------------------------------------------------------------
# 4. Model comparison: log-likelihood, parameters, AIC, BIC, likelihood-ratio tests
# ---------------------------------------------------------------------------

llf_lin = float(fits["linear"][0].llf)
k_lin = fits["linear"][2]
aic_lin = fits["linear"][3]
bic_lin = fits["linear"][4]
for m in MODELS:
    res, design, k, aic, bic = fits[m]
    put(f"a4.model.{m}.llf", float(res.llf))
    put(f"a4.model.{m}.parameters_n", k)
    put(f"a4.model.{m}.year_parameters_n", YEAR_PARAMETERS[m])
    put(f"a4.model.{m}.aic", aic)
    put(f"a4.model.{m}.bic", bic)
    put(f"a4.model.{m}.aic_minus_linear", aic - aic_lin)
    put(f"a4.model.{m}.bic_minus_linear", bic - bic_lin)
    if k - fits["linear"][2] != YEAR_PARAMETERS[m] - 1:
        raise R1.R1Error(f"{m}: parameter count does not match the year parameters")


def lr_test(big, small):
    llf_big, llf_small = float(fits[big][0].llf), float(fits[small][0].llf)
    df_diff = fits[big][2] - fits[small][2]
    stat = 2.0 * (llf_big - llf_small)
    if stat < -1e-6:
        raise R1.R1Error(f"{big} has a lower likelihood than the nested {small}")
    stat = max(stat, 0.0)
    p = float(stats.chi2.sf(stat, df_diff))
    log10_p = float(stats.chi2.logsf(stat, df_diff) / math.log(10.0))
    return stat, df_diff, p, log10_p


LR = {}
for big in ("categorical", "quadratic", "spline4", "spline5"):
    stat, df_diff, p, log10_p = lr_test(big, "linear")
    LR[big] = (stat, df_diff, p, log10_p)
    put(f"a4.lr.{big}_vs_linear.statistic", stat)
    put(f"a4.lr.{big}_vs_linear.df", df_diff)
    put(f"a4.lr.{big}_vs_linear.p", p)
    put(f"a4.lr.{big}_vs_linear.log10_p", log10_p)
# Extra context: does a smooth curve capture what the categorical model sees?
for small in ("quadratic", "spline4", "spline5"):
    stat, df_diff, p, log10_p = lr_test("categorical", small)
    put(f"a4.lr.categorical_vs_{small}.statistic", stat)
    put(f"a4.lr.categorical_vs_{small}.df", df_diff)
    put(f"a4.lr.categorical_vs_{small}.p", p)
    put(f"a4.lr.categorical_vs_{small}.log10_p", log10_p)
stat45, df45, p45, log45 = lr_test("spline5", "spline4")
put("a4.lr.spline5_vs_spline4.statistic", stat45)
put("a4.lr.spline5_vs_spline4.df", df45)
put("a4.lr.spline5_vs_spline4.p", p45)
put("a4.lr.spline5_vs_spline4.log10_p", log45)
put("a4.lr.sensitivity_conclusion_unchanged",
    bool((LR["spline4"][2] < 0.05) == (LR["spline5"][2] < 0.05)))

# The spline knots, and the proof that the linear year term is nested inside each spline model.
put("a4.spline4.knots", knots4)
put("a4.spline5.knots", knots5)
put("a4.spline4.boundary_knots", [knots4[0], knots4[-1]])
put("a4.spline5.boundary_knots", [knots5[0], knots5[-1]])
for m, basis in (("spline4", basis4), ("spline5", basis5)):
    design_matrix = np.column_stack([np.ones(n), basis])
    target = x_year - YEAR0
    coef, *_ = np.linalg.lstsq(design_matrix, target, rcond=None)
    max_resid = float(np.abs(design_matrix @ coef - target).max())
    if max_resid > 1e-8:
        raise R1.R1Error(f"linear year is not inside the {m} span (max residual {max_resid})")
    put(f"a4.{m}.linear_nested_max_residual", max_resid)

# Quadratic year terms (year terms only; the covariates are never read out).
quad_res = fits["quadratic"][0]
b1 = float(quad_res.params["YEAR.centred"])
b2 = float(quad_res.params["YEAR.centred squared"])
se2 = float(quad_res.bse["YEAR.centred squared"])
p2 = float(quad_res.pvalues["YEAR.centred squared"])
put("a4.quadratic.centre_year", CENTRE_YEAR)
put("a4.quadratic.linear_term_beta", b1)
put("a4.quadratic.squared_term_beta", b2)
put("a4.quadratic.squared_term_se", se2)
put("a4.quadratic.squared_term_p", p2)
put("a4.quadratic.squared_term_significant", bool(p2 < 0.05))
if p2 < 0.05 and b2 != 0.0:
    vertex = CENTRE_YEAR - b1 / (2.0 * b2)
    cov = quad_res.cov_params().loc[["YEAR.centred", "YEAR.centred squared"],
                                    ["YEAR.centred", "YEAR.centred squared"]].to_numpy()
    grad = np.array([-1.0 / (2.0 * b2), b1 / (2.0 * b2 * b2)])
    se_vertex = math.sqrt(float(grad @ cov @ grad))
    put("a4.quadratic.vertex_year", vertex)
    put("a4.quadratic.vertex_year_lo", vertex - Z95 * se_vertex)
    put("a4.quadratic.vertex_year_hi", vertex + Z95 * se_vertex)
    put("a4.quadratic.opens_downward", bool(b2 < 0))

# ---------------------------------------------------------------------------
# 5. Observed and fitted rates by year
# ---------------------------------------------------------------------------


def wilson(x, total):
    p_hat = x / total
    denom = 1.0 + Z95 ** 2 / total
    centre = (p_hat + Z95 ** 2 / (2.0 * total)) / denom
    half = Z95 * math.sqrt(p_hat * (1.0 - p_hat) / total + Z95 ** 2 / (4.0 * total * total)) / denom
    return centre - half, centre + half


def fitted_matched(kind):
    """Fitted same-day rate averaged over the patients actually observed in each year."""
    res, design = fits[kind][0], fits[kind][1]
    p_hat = np.asarray(res.predict(design), dtype=float)
    return {yr: float(p_hat[year == yr].mean()) for yr in R1.YEARS}


def fitted_standardized(kind):
    """Fitted rate for every patient's own covariates with the year set to that year."""
    res = fits[kind][0]
    out = {}
    for yr in R1.YEARS:
        design = full_design(kind, np.full(n, yr))
        out[yr] = float(np.asarray(res.predict(design), dtype=float).mean())
    return out


matched = {k: fitted_matched(k) for k in ("linear", "categorical")}
standardized = {k: fitted_standardized(k) for k in ("linear", "categorical")}

cat_res = fits["categorical"][0]
cat_ci = cat_res.conf_int()
rows_year = []
obs = {}
for yr in R1.YEARS:
    in_year = year == yr
    n_y = int(in_year.sum())
    x_y = int(y[in_year].sum())
    lo_w, hi_w = wilson(x_y, n_y)
    obs[yr] = (n_y, x_y, x_y / n_y, lo_w, hi_w)
    # The categorical model has a year dummy for every year but 2012 and an intercept, so its
    # score equations force the case-mix-matched fitted rate of EVERY year to equal the observed rate.
    if abs(matched["categorical"][yr] - x_y / n_y) > 1e-8:
        raise R1.R1Error(f"categorical matched fit differs from the observed rate in {yr}")
put("a4.categorical.matched_equals_observed", True)

for yr in R1.YEARS:
    n_y, x_y, rate, lo_w, hi_w = obs[yr]
    pre = f"a4.year.{yr}"
    put(f"{pre}.n", n_y)
    put(f"{pre}.events_n", x_y)
    put(f"{pre}.observed_pct", 100.0 * rate)
    put(f"{pre}.wilson_lo_pct", 100.0 * lo_w)
    put(f"{pre}.wilson_hi_pct", 100.0 * hi_w)
    put(f"{pre}.table2_n", int(t2[yr][1]))
    put(f"{pre}.table2_events_n", int(t2[yr][0]))
    put(f"{pre}.table2_pct", 100.0 * t2[yr][0] / t2[yr][1])
    put(f"{pre}.linear_matched_pct", 100.0 * matched["linear"][yr])
    put(f"{pre}.linear_standardized_pct", 100.0 * standardized["linear"][yr])
    put(f"{pre}.categorical_matched_pct", 100.0 * matched["categorical"][yr])
    put(f"{pre}.categorical_standardized_pct", 100.0 * standardized["categorical"][yr])
    put(f"{pre}.gap_linear_matched_pp", 100.0 * (rate - matched["linear"][yr]))
    put(f"{pre}.gap_linear_standardized_pp", 100.0 * (rate - standardized["linear"][yr]))
    put(f"{pre}.linear_matched_inside_wilson", bool(lo_w <= matched["linear"][yr] <= hi_w))
    put(f"{pre}.linear_implied_or", lin_or ** (yr - YEAR0))
    if yr == YEAR0:
        put(f"{pre}.categorical_or", 1.0)
    else:
        name = f"YEAR.{yr}"
        put(f"{pre}.categorical_or", math.exp(float(cat_res.params[name])))
        put(f"{pre}.categorical_or_lo", math.exp(float(cat_ci.loc[name, 0])))
        put(f"{pre}.categorical_or_hi", math.exp(float(cat_ci.loc[name, 1])))
        put(f"{pre}.categorical_or_p", float(cat_res.pvalues[name]))
        put(f"{pre}.linear_implied_or_inside_categorical_ci",
            bool(math.exp(float(cat_ci.loc[name, 0])) <= lin_or ** (yr - YEAR0)
                 <= math.exp(float(cat_ci.loc[name, 1]))))

peak_year = max(R1.YEARS, key=lambda yr: obs[yr][2])
put("a4.peak.year", peak_year)
put("a4.peak.observed_pct", 100.0 * obs[peak_year][2])
put("a4.peak.table2_pct", 100.0 * t2[peak_year][0] / t2[peak_year][1])

# Late-period contrasts between year terms of the categorical model (year terms only).
cov_cat = cat_res.cov_params()


def year_contrast(later, earlier):
    """Odds ratio of year `later` against year `earlier` from the categorical model."""
    def coef(yr):
        return 0.0 if yr == YEAR0 else float(cat_res.params[f"YEAR.{yr}"])
    weights = {later: 1.0, earlier: -1.0}
    est = sum(weights[yr] * coef(yr) for yr in weights)
    names = [f"YEAR.{yr}" for yr in weights if yr != YEAR0]
    w = np.array([weights[yr] for yr in weights if yr != YEAR0])
    var = float(w @ cov_cat.loc[names, names].to_numpy() @ w)
    se_c = math.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(est) / se_c))
    return math.exp(est), math.exp(est - Z95 * se_c), math.exp(est + Z95 * se_c), p


for later, earlier in ((2021, 2020), (2022, 2021), (2023, 2022), (2024, 2023), (2024, 2022), (2024, 2021)):
    o, lo_c, hi_c, p_c = year_contrast(later, earlier)
    put(f"a4.contrast.{later}_vs_{earlier}.or", o)
    put(f"a4.contrast.{later}_vs_{earlier}.or_lo", lo_c)
    put(f"a4.contrast.{later}_vs_{earlier}.or_hi", hi_c)
    put(f"a4.contrast.{later}_vs_{earlier}.p", p_c)

# ---------------------------------------------------------------------------
# 6. Tables S8A and S8B
# ---------------------------------------------------------------------------


def f1(v):
    return f"{v:,.1f}"


header_a = ["Model", "Year parameters", "Total parameters", "Log-likelihood", "AIC", "BIC",
            "AIC minus linear model", "BIC minus linear model",
            "Likelihood-ratio statistic against linear model", "Degrees of freedom",
            "P value against linear model", "Patients", "Same-day discharges"]
rows_a = []
for m in MODELS:
    res, design, k, aic, bic = fits[m]
    if m == "linear":
        lr_cells = ["Reference", "Reference", "Reference"]
    else:
        stat, df_diff, p, _ = LR[m]
        lr_cells = [f1(stat), str(df_diff), VD.fmt_p(p, spaced=False)]
    rows_a.append([LABELS[m], str(YEAR_PARAMETERS[m]), str(k), f1(float(res.llf)), f1(aic), f1(bic),
                   f1(aic - aic_lin), f1(bic - bic_lin)] + lr_cells + [VD.fmt_count(n), VD.fmt_count(int(y.sum()))])
R1.write_csv_table(R1.TABLES_OUT / "tableS8_year_trend_models.csv", header_a, rows_a)

header_b = [
    "Year", "Patients", "Same-day discharges",
    "Observed same-day rate, %", "Observed rate, Wilson 95% confidence interval, %",
    "Linear model fitted rate, case-mix matched, %", "Linear model fitted rate, standardized, %",
    "Observed minus linear fit, case-mix matched, percentage points",
    "Categorical model fitted rate, case-mix matched, %", "Categorical model fitted rate, standardized, %",
    "Adjusted odds ratio against 2012, categorical model (95% confidence interval)",
    "Odds ratio against 2012 implied by the linear model",
    "Patients with a recorded stay (Table 2 basis)", "Same-day discharges (Table 2 basis)",
    "Same-day rate (Table 2 basis), %",
]
rows_b = []
for yr in R1.YEARS:
    n_y, x_y, rate, lo_w, hi_w = obs[yr]
    if yr == YEAR0:
        or_text = "1 (reference)"
    else:
        name = f"YEAR.{yr}"
        or_text = VD.fmt_or_ci(math.exp(float(cat_res.params[name])), math.exp(float(cat_ci.loc[name, 0])),
                               math.exp(float(cat_ci.loc[name, 1])), 2)
    rows_b.append([
        str(yr), VD.fmt_count(n_y), VD.fmt_count(x_y),
        f"{100 * rate:.1f}", VD.fmt_ci(100 * lo_w, 100 * hi_w, 1),
        f"{100 * matched['linear'][yr]:.1f}", f"{100 * standardized['linear'][yr]:.1f}",
        f"{100 * (rate - matched['linear'][yr]):.1f}",
        f"{100 * matched['categorical'][yr]:.1f}", f"{100 * standardized['categorical'][yr]:.1f}",
        or_text, f"{lin_or ** (yr - YEAR0):.2f}",
        VD.fmt_count(t2[yr][1]), VD.fmt_count(t2[yr][0]), f"{100 * t2[yr][0] / t2[yr][1]:.1f}",
    ])
R1.write_csv_table(R1.TABLES_OUT / "tableS8_year_by_year.csv", header_b, rows_b)

# ---------------------------------------------------------------------------
# 7. Figure S4 (no title): A observed versus fitted rates, B odds ratios against 2012
# ---------------------------------------------------------------------------

written = []
if R1.FIGURES:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    R1.style_figure(8.0)
    yrs = np.array(R1.YEARS)
    obs_pct = np.array([100 * obs[yr][2] for yr in R1.YEARS])
    lo_pct = np.array([100 * obs[yr][3] for yr in R1.YEARS])
    hi_pct = np.array([100 * obs[yr][4] for yr in R1.YEARS])
    lin_matched_pct = np.array([100 * matched["linear"][yr] for yr in R1.YEARS])
    cat_matched_pct = np.array([100 * matched["categorical"][yr] for yr in R1.YEARS])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.32))

    ax_a.plot(yrs, lin_matched_pct, "-", color=R1.ORANGE, lw=1.6, zorder=2)
    ax_a.errorbar(yrs, obs_pct, yerr=[obs_pct - lo_pct, hi_pct - obs_pct], fmt="o", color=R1.INK, ms=3.8,
                  elinewidth=0.9, capsize=2.0, zorder=4)
    ax_a.plot(yrs, cat_matched_pct, "D", mfc="none", mec=R1.BLUE, mew=1.1, ms=6.6, zorder=3)
    ax_a.set_xlim(2011.4, 2024.6)
    ax_a.set_ylim(0, max(float(lin_matched_pct.max()), float(hi_pct.max())) * 1.12)
    ax_a.set_xticks(list(range(2012, 2025, 2)))
    ax_a.set_xlabel("Year of surgery")
    ax_a.set_ylabel("Same-day discharge, %")
    ax_a.grid(axis="y", lw=0.5)
    ax_a.text(2011.7, ax_a.get_ylim()[1] * 0.97, "A", fontsize=10, fontweight="bold", va="top", color=R1.INK)
    ax_a.text(2012.15, ax_a.get_ylim()[1] * 0.80,
              "Linear-year model fit\n(case-mix matched)", color=R1.ORANGE, fontsize=7.5, va="top", ha="left")
    ax_a.text(2017.5, ax_a.get_ylim()[1] * 0.36,
              "Observed rate with\nWilson 95% interval (points)", color=R1.INK, fontsize=7.5, va="top", ha="left")
    ax_a.text(2017.5, ax_a.get_ylim()[1] * 0.19,
              "Categorical-year model fit\n(open diamonds)", color=R1.BLUE, fontsize=7.5, va="top", ha="left")

    cat_or = np.array([1.0] + [math.exp(float(cat_res.params[f"YEAR.{yr}"])) for yr in R1.YEARS[1:]])
    cat_or_lo = np.array([1.0] + [math.exp(float(cat_ci.loc[f"YEAR.{yr}", 0])) for yr in R1.YEARS[1:]])
    cat_or_hi = np.array([1.0] + [math.exp(float(cat_ci.loc[f"YEAR.{yr}", 1])) for yr in R1.YEARS[1:]])
    implied = lin_or ** (yrs - YEAR0)
    ax_b.plot(yrs, implied, "-", color=R1.ORANGE, lw=1.6, zorder=2)
    ax_b.errorbar(yrs, cat_or, yerr=[cat_or - cat_or_lo, cat_or_hi - cat_or], fmt="D", color=R1.BLUE, mfc="none",
                  mew=1.1, ms=5.0, elinewidth=0.9, capsize=2.0, zorder=4)
    ax_b.set_yscale("log")
    upper = float(max(implied.max(), cat_or_hi.max())) * 1.15
    # lower limit from the data: every lower confidence limit stays visible, rounded down to a multiple of 0.05
    lower = math.floor(float(min(cat_or_lo.min(), implied.min())) * 0.95 / 0.05) * 0.05
    lower = round(lower, 2)
    ax_b.set_ylim(lower, upper)
    ticks = [t for t in (1, 2, 3, 4, 5, 6, 8) if t <= upper]
    ax_b.yaxis.set_major_locator(FixedLocator(ticks))
    ax_b.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax_b.yaxis.set_minor_locator(NullLocator())
    ax_b.set_xlim(2011.4, 2024.6)
    ax_b.set_xticks(list(range(2012, 2025, 2)))
    ax_b.set_xlabel("Year of surgery")
    ax_b.set_ylabel("Adjusted odds ratio against 2012 (log scale)")
    ax_b.grid(axis="y", lw=0.5)
    ax_b.axhline(1.0, color=R1.RULE, lw=0.8, zorder=1)
    ax_b.text(2011.7, upper * 0.97, "B", fontsize=10, fontweight="bold", va="top", color=R1.INK)
    ax_b.text(2012.15, upper * 0.80, "Implied by the linear-year model\n(odds ratio per year, compounded)",
              color=R1.ORANGE, fontsize=7.5, va="top", ha="left")
    ax_b.text(2018.6, 1.55, "Categorical-year model\n(points, 95% confidence interval)", color=R1.BLUE,
              fontsize=7.5, va="top", ha="left")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.97, bottom=0.15)
    written = R1.save_png(fig, R1.FIGURES_OUT / "figS4_year_trend_diagnostics.png")

# ---------------------------------------------------------------------------
# 8. The fragment
# ---------------------------------------------------------------------------

n_keys = R1.write_fragment(R1.FRAGMENTS_OUT / "results_a4.json", R, PREFIX)

print(f"[a4] n = {n:,}, same-day events = {int(y.sum()):,} ({100 * y.mean():.3f}%); excluded {n_all - n}")
print(f"[a4] linear OR per year {lin_or:.6f} ({lin_lo:.6f} to {lin_hi:.6f}); v1 refit identical")
for m in MODELS:
    res, design, k, aic, bic = fits[m]
    line = f"[a4] {m:12s} k={k:2d} llf={float(res.llf):14.3f} AIC={aic:14.3f} BIC={bic:14.3f}"
    if m in LR:
        line += f"  LR={LR[m][0]:10.3f} df={LR[m][1]:2d} p={LR[m][2]:.3g} log10p={LR[m][3]:.1f}"
    print(line)
print(f"[a4] spline4 knots {knots4}; spline5 knots {knots5}")
print(f"[a4] quadratic squared term beta {b2:.5f}, P {p2:.3g}; vertex "
      f"{R.get('a4.quadratic.vertex_year', float('nan')):.2f}")
for yr in R1.YEARS:
    n_y, x_y, rate, lo_w, hi_w = obs[yr]
    print(f"[a4] {yr} obs {100 * rate:6.2f}% ({x_y}/{n_y}) Wilson {100 * lo_w:5.2f}-{100 * hi_w:5.2f} | "
          f"lin matched {100 * matched['linear'][yr]:6.2f} std {100 * standardized['linear'][yr]:6.2f} | "
          f"cat matched {100 * matched['categorical'][yr]:6.2f} std {100 * standardized['categorical'][yr]:6.2f} | "
          f"cat OR {R[f'a4.year.{yr}.categorical_or']:.3f} lin OR {lin_or ** (yr - YEAR0):.3f}")
print(f"[a4] wrote {n_keys} keys to results_a4.json; figure files: {[p.name for p in written]}")
