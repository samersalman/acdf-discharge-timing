#!/usr/bin/env python3
"""a5_direct_standardization.py - A5 adjusted strata by direct standardization.

Question. Do the subgroup differences survive adjustment for the other stratifiers? Table 5
compares the levels of each locked stratifier (age group, ASA class, fusion levels, sex) CRUDE.
Here each level is compared after standardizing it to the pooled distribution of the OTHER THREE
stratifiers, so a level difference in residual-risk timing cannot be produced by the other three
stratifiers' mix.

Method (the v1 estimator, on weighted daily counts; nothing is fitted).
  Cells: the cross-classification of the other three stratifiers (12 to 24 cells for each
  stratifier S). Patients with sex neither male nor female (35) are excluded from A5 entirely.
  Positivity: a cell is dropped from the standard population when ANY level of S has fewer than
  20 patients in it. The standard population is the pooled retained cells: N_c patients in
  retained cell c, N = sum of N_c. For a patient in level l and cell c (n_lc patients, level
  total n_l over the retained cells) the weight is w = (N_c / N) / (n_lc / n_l), so the weighted
  cell composition inside EVERY level equals the pooled one. The weighted daily counts are
  e1w(d) = n_l * sum_c (N_c / N) * e1_lc(d) / n_lc and likewise e2w(d) for the competing death,
  with n_w = n_l; they go through the v1 W.aalen_johansen, W.conditional_risk_aj and
  W.first_day_below. The crude comparison is recomputed on the SAME retained patients (so crude
  and standardized are like for like) and v1's full-cohort Table 5 value is carried beside it.
  Bootstrap: 1,000 replicates, np.random.seed(VD.SEED) once, levels visited in stratifier then
  level order. Each level's retained patients are resampled multinomially over the cells of
  (other-cell x outcome cell), the outcome cell being event on day d, death on day d, or event
  free, so n_lc, e1_lc and e2_lc vary and N_c and the weights are recomputed every replicate.
  A retained cell that is empty in some level of a replicate is dropped for that level and the
  weights renormalised over its remaining cells; the replicates where that happens are counted.
  Suppression: a level with fewer than 10 composite events (unweighted) is withheld.

Scope: no coefficient, no odds ratio, no discrimination statistic.

Sanity tests run BEFORE any bootstrap draw (all must pass or the run stops):
  (i)  a null standardization with a single cell reproduces the crude v1 estimates exactly;
  (ii) standardizing a level to its OWN composition reproduces its crude curve.

Outputs (under the results directory): fragments/results_a5.json (namespace a5.),
tables-csv/tableS9_adjusted_strata.csv.
Run:  PYTHONDONTWRITEBYTECODE=1 python3 a5_direct_standardization.py
"""

import sys; sys.dont_write_bytecode = True
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import math

import numpy as np

import r1_common as R1

VD, W = R1.VD, R1.W
R1.seed_all()

PREFIX = ("a5.",)
MIN_CELL = 20                       # positivity rule: every level of S needs at least 20 patients in a cell
B = W.BOOTSTRAP_REPLICATES          # 1,000
DAYS = R1.DAYS
THRESHOLDS = (0.005, 0.0025)        # the two thresholds the comparison is read at
COLUMNS = list(VD.STRATIFIER_COLUMNS)           # AGE_CAT, ASA_CAT, LEVELS_CAT, MALE
SPEC = {sp.column: sp for sp in VD.STRATIFIERS}

R = {}


def put(key, value):
    if key in R:
        raise R1.R1Error(f"duplicate key {key}")
    R[key] = value


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

df = R1.load_cohort()
events = R1.make_events(df, "WINDOW_COMP", "POD_WINDOW")
if events.n_events != 1015:
    raise R1.R1Error(f"composite events {events.n_events}, expected 1,015")

# v1 crude, full cohort, with v1's own bootstrap intervals (reseeds with VD.SEED; the A5 stream is
# seeded again below, after this call and after the sanity tests).
v1_fits = R1.fit_overall_and_strata(df, events, bootstrap=True)
v1_by_slug = R1.index_by_slug(v1_fits)

keep = df["MALE"].notna().to_numpy()
n_pop = int(keep.sum())
n_nonbinary = int((~keep).sum())
if n_nonbinary != 35 or n_pop != 112_607:
    raise R1.R1Error(f"binary-sex patients {n_pop}, non-binary {n_nonbinary}; expected 112,607 and 35")
put("a5.population.n", n_pop)
put("a5.population.events_n", int((keep & events.is_event).sum()))
put("a5.population.excluded_non_binary_n", n_nonbinary)
put("a5.population.excluded_non_binary_events_n", int((~keep & events.is_event).sum()))
put("a5.population.cell_size_rule_min_patients_n", MIN_CELL)
put("a5.bootstrap.replicates_n", B)

codes = {}
for column in COLUMNS:
    text = df[column].astype("string")
    code = np.full(len(df), -1, dtype=int)
    for i, raw in enumerate(SPEC[column].levels):
        code[(text == raw).fillna(False).to_numpy()] = i
    if not (code[keep] >= 0).all():
        raise R1.R1Error(f"{column}: a patient with male or female sex has no level")
    codes[column] = code

pod_event = events.pod_event
pod_death = events.pod_death


def stratifier_slug(column):
    return W.STRATIFIER_SLUG[column]


def level_slug(column, raw):
    return W.level_slug(column, raw)


# ---------------------------------------------------------------------------
# 2. Cell counts for one stratifier S: n_lc, e1_lc(d), e2_lc(d) for every level and every cell
# ---------------------------------------------------------------------------


class Layout:
    """Cells of the other three stratifiers for stratifier `column`, and the level-by-cell counts."""

    def __init__(self, column):
        self.column = column
        self.levels = list(SPEC[column].levels)
        self.others = [c for c in COLUMNS if c != column]
        radix = [len(SPEC[c].levels) for c in self.others]
        self.n_cells = int(np.prod(radix))
        cell = np.zeros(len(df), dtype=int)
        for c, r in zip(self.others, radix):
            cell = cell * r + np.where(codes[c] >= 0, codes[c], 0)
        self.cell = cell
        self.radix = radix
        L = len(self.levels)
        self.n_lc = np.zeros((L, self.n_cells))
        self.e1 = np.zeros((L, self.n_cells, DAYS))
        self.e2 = np.zeros((L, self.n_cells, DAYS))
        for i in range(L):
            mask = keep & (codes[column] == i)
            self.n_lc[i] = np.bincount(cell[mask], minlength=self.n_cells)
            ev = mask & events.is_event
            cp = mask & events.is_competing
            self.e1[i] = np.bincount(cell[ev] * DAYS + pod_event[ev].astype(int),
                                     minlength=self.n_cells * DAYS).reshape(self.n_cells, DAYS)
            self.e2[i] = np.bincount(cell[cp] * DAYS + pod_death[cp].astype(int),
                                     minlength=self.n_cells * DAYS).reshape(self.n_cells, DAYS)
        self.retained = self.n_lc.min(axis=0) >= MIN_CELL
        if self.n_lc.sum() != n_pop:
            raise R1.R1Error(f"{column}: the level-by-cell table does not cover {n_pop:,} patients")

    def cell_label(self, c):
        """Reader-facing text for cell c: the levels of the other three stratifiers, in column order."""
        parts = []
        rem = c
        indices = []
        for r in reversed(self.radix):
            indices.append(rem % r)
            rem //= r
        indices.reverse()
        for column, idx in zip(self.others, indices):
            parts.append(VD.pretty_level(column, SPEC[column].levels[idx]))
        return ", ".join(parts)


LAYOUTS = {c: Layout(c) for c in COLUMNS}


# ---------------------------------------------------------------------------
# 3. The standardized estimator on weighted daily counts
# ---------------------------------------------------------------------------


def standardized_counts(n_lc, e1, e2, l, own=False):
    """Weighted daily counts for level l.

    n_lc (L, Bn, C) patients by level, replicate, cell; e1, e2 (L, Bn, C, D) daily counts.
    omega_c = N_c / N with N_c pooled over the levels (or n_lc / n_l when own=True, the level's
    own composition). A cell empty in level l of a replicate gets weight zero and omega is
    renormalised over the cells that remain. Returns e1w, e2w (Bn, D), the per-patient weights
    (Bn, C) and the level size n_l (Bn,). Algebraically e1w(d) = n_l * sum_c omega_c e1_lc(d) / n_lc,
    written as sum_c [omega_c * n_l / n_lc] e1_lc(d) so a single cell gives a weight of exactly 1.
    """
    n_level = n_lc[l]
    n_l = n_level.sum(axis=1)
    nonempty = n_level > 0
    base = n_level if own else n_lc.sum(axis=0)
    omega = np.where(nonempty, base, 0.0)
    omega = omega / omega.sum(axis=1, keepdims=True)
    safe = np.where(nonempty, n_level, 1.0)
    weight = np.where(nonempty, omega * n_l[:, None] / safe, 0.0)
    e1w = np.einsum("bc,bcd->bd", weight, e1[l])
    e2w = np.einsum("bc,bcd->bd", weight, e2[l])
    return e1w, e2w, weight, n_l


def curves_from_counts(e1w, e2w, n_l):
    """v1 estimator on (Bn, D) counts: returns cif (Bn, 32) and the conditional-risk curves (Bn, 31)."""
    cif, surv, _ = W.aalen_johansen(e1w, e2w, np.asarray(n_l, dtype=float))
    return cif, W.conditional_risk_aj(cif, surv)


# ---------------------------------------------------------------------------
# 4. Sanity tests (before any random draw)
# ---------------------------------------------------------------------------

sanity = {"null_cells_checked_n": 0, "null_max_abs_diff": 0.0, "own_cells_checked_n": 0,
          "own_max_abs_diff": 0.0, "own_v1_checked_n": 0}

# (i) one cell, weights exactly 1: every non-suppressed v1 stratum, including the overall row.
for fit in v1_fits:
    if fit.suppressed:
        continue
    n_s = fit.n
    e1c = fit.e1[None, None, None, :]                       # (L=1, Bn=1, C=1, D)
    e2c = fit.e2[None, None, None, :]
    n_one = np.array([[[float(n_s)]]])
    e1w, e2w, weight, n_l = standardized_counts(n_one, e1c[:, :, :, :].reshape(1, 1, 1, DAYS),
                                                e2c.reshape(1, 1, 1, DAYS), 0)
    if weight.shape != (1, 1) or weight[0, 0] != 1.0:
        raise R1.R1Error("null standardization: the single-cell weight is not exactly 1")
    cif_w, curve_w = curves_from_counts(e1w, e2w, n_l)
    diff = float(np.abs(curve_w[0] - fit.curve).max())
    if not np.array_equal(curve_w[0], fit.curve) or abs(cif_w[0, -1] - fit.cif30) != 0.0:
        raise R1.R1Error(f"SANITY TEST (i) FAILED on {fit.slug}: single-cell standardization differs "
                         f"from the crude v1 curve (max abs difference {diff:.3g})")
    for t in R1.R1_THRESHOLDS:
        if R1.threshold_day(curve_w[0], t) != fit.threshold_day(t):
            raise R1.R1Error(f"SANITY TEST (i) FAILED on {fit.slug}: threshold day at {t} differs")
    sanity["null_cells_checked_n"] += 1
    sanity["null_max_abs_diff"] = max(sanity["null_max_abs_diff"], diff)

# (ii) own composition, every analysed level of every stratifier, all cells (no positivity drop).
for column in COLUMNS:
    lay = LAYOUTS[column]
    for i, raw in enumerate(lay.levels):
        n_level = lay.n_lc[i]
        mask = keep & (codes[column] == i)
        e1_crude, e2_crude = W.counts_by_day(pod_event[mask & events.is_event],
                                             pod_death[mask & events.is_competing])
        n_i = float(mask.sum())
        cif_c, surv_c, _ = W.aalen_johansen(e1_crude[None, :], e2_crude[None, :], np.array([n_i]))
        curve_c = W.conditional_risk_aj(cif_c, surv_c)[0]
        e1w, e2w, weight, n_l = standardized_counts(lay.n_lc[:, None, :], lay.e1[:, None, :, :],
                                                    lay.e2[:, None, :, :], i, own=True)
        cif_w, curve_w = curves_from_counts(e1w, e2w, n_l)
        diff = float(np.abs(curve_w[0] - curve_c).max())
        if diff > 1e-12 or abs(cif_w[0, -1] - cif_c[0, -1]) > 1e-12:
            raise R1.R1Error(f"SANITY TEST (ii) FAILED on {column} level {raw}: max abs difference {diff:.3g}")
        for t in R1.R1_THRESHOLDS:
            if R1.threshold_day(curve_w[0], t) != R1.threshold_day(curve_c, t):
                raise R1.R1Error(f"SANITY TEST (ii) FAILED on {column} level {raw}: day at {t} differs")
        sanity["own_cells_checked_n"] += 1
        sanity["own_max_abs_diff"] = max(sanity["own_max_abs_diff"], diff)
        if column == "MALE":
            v1_fit = v1_by_slug[f"sex.{level_slug(column, raw)}"]
            if not np.allclose(curve_w[0], v1_fit.curve, atol=1e-12, rtol=0.0):
                raise R1.R1Error(f"SANITY TEST (ii) FAILED against v1 on the {raw} sex level")
            sanity["own_v1_checked_n"] += 1

for key, value in sanity.items():
    put(f"a5.sanity.{key}", value)
put("a5.sanity.null_standardization_passes", True)
put("a5.sanity.own_composition_passes", True)

# ---------------------------------------------------------------------------
# 5. Per stratifier: positivity, point estimates, bootstrap (RNG stream starts here)
# ---------------------------------------------------------------------------

np.random.seed(VD.SEED)

LEVEL_RESULTS = []          # one dict per analysed level, in stratifier then level order
STRAT_RESULTS = []          # one dict per stratifier
dropped_union = np.zeros(len(df), dtype=bool)
dropped_all = np.ones(len(df), dtype=bool)


def threshold_stats(curve_b, cif_b, point_curve, point_cif30):
    """Point estimates and bootstrap intervals for one estimate (crude or standardized)."""
    out = {"cif30": float(point_cif30)}
    lo, hi = W.percentile_ci(cif_b[:, -1])
    out["cif30_lo"], out["cif30_hi"] = float(lo), float(hi)
    for t in THRESHOLDS:
        day = R1.threshold_day(point_curve, t)
        boot_days = R1.threshold_day(curve_b, t)
        d_lo, d_hi = W.percentile_ci(boot_days.astype(float))
        out[t] = {
            "day": int(day),
            "lo": int(math.floor(d_lo)),
            "hi": int(math.ceil(d_hi)),
            "stays_below": bool(R1.stays_below_from(point_curve, t, day)),
            "boot_days": boot_days,
        }
    return out


for column in COLUMNS:
    lay = LAYOUTS[column]
    skey = stratifier_slug(column)
    keep_cells = lay.retained
    c_idx = np.where(keep_cells)[0]
    Cr = len(c_idx)
    L = len(lay.levels)
    dropped_cells = np.where(~keep_cells)[0]
    in_dropped = keep & np.isin(lay.cell, dropped_cells)
    dropped_union |= in_dropped
    dropped_all &= in_dropped
    n_dropped = int(in_dropped.sum())
    n_events_dropped = int((in_dropped & events.is_event).sum())
    put(f"a5.{skey}.cells_n", lay.n_cells)
    smallest_by_cell = lay.n_lc.min(axis=0)                 # the smallest level in each cell
    c_small = int(np.argmin(smallest_by_cell))
    l_small = int(np.argmin(lay.n_lc[:, c_small]))
    put(f"a5.{skey}.smallest_cell_level_n", int(smallest_by_cell[c_small]))
    put(f"a5.{skey}.smallest_cell_label", f"{VD.pretty_level(column, lay.levels[l_small])}, {lay.cell_label(c_small)}")
    put(f"a5.{skey}.cells_with_a_level_below_50_n", int((smallest_by_cell < 50).sum()))
    put(f"a5.{skey}.retained_cells_n", Cr)
    put(f"a5.{skey}.dropped_cells_n", int(len(dropped_cells)))
    put(f"a5.{skey}.dropped_patients_n", n_dropped)
    put(f"a5.{skey}.dropped_share_pct", 100.0 * n_dropped / n_pop)
    put(f"a5.{skey}.dropped_events_n", n_events_dropped)
    put(f"a5.{skey}.retained_patients_n", n_pop - n_dropped)
    put(f"a5.{skey}.retained_share_pct", 100.0 * (n_pop - n_dropped) / n_pop)
    if len(dropped_cells):
        put(f"a5.{skey}.dropped_cells", [lay.cell_label(int(c)) for c in dropped_cells])
        put(f"a5.{skey}.dropped_cells_smallest_level_n",
            [int(lay.n_lc[:, c].min()) for c in dropped_cells])
    strat_info = {"column": column, "skey": skey, "dropped_n": n_dropped,
                  "dropped_cells": [lay.cell_label(int(c)) for c in dropped_cells]}
    STRAT_RESULTS.append(strat_info)

    # retained-cell arrays: point estimates (Bn = 1) and bootstrap draws (Bn = B)
    n_pt = lay.n_lc[:, keep_cells][:, None, :]                         # (L, 1, Cr)
    e1_pt = lay.e1[:, keep_cells, :][:, None, :, :]                    # (L, 1, Cr, D)
    e2_pt = lay.e2[:, keep_cells, :][:, None, :, :]
    n_level_total = n_pt[:, 0, :].sum(axis=1)                          # (L,) retained patients of each level
    events_ret = e1_pt[:, 0].sum(axis=(1, 2))                          # (L,) composite events among retained
    suppressed = events_ret < VD.MIN_STRATUM_EVENTS

    n_bs = np.zeros((L, B, Cr))
    e1_bs = np.zeros((L, B, Cr, DAYS))
    e2_bs = np.zeros((L, B, Cr, DAYS))
    for i in range(L):                                                  # level order: the RNG order
        if suppressed[i]:                                               # no draw; point counts held fixed
            n_bs[i] = n_pt[i]
            e1_bs[i] = e1_pt[i]
            e2_bs[i] = e2_pt[i]
            continue
        a1 = e1_pt[i, 0]                                                # (Cr, D)
        a2 = e2_pt[i, 0]
        free = n_pt[i, 0] - a1.sum(axis=1) - a2.sum(axis=1)
        if (free < 0).any():
            raise R1.R1Error(f"{column} level {lay.levels[i]}: more events than patients in a cell")
        cells = np.concatenate([a1.ravel(), a2.ravel(), free])
        probs = cells / cells.sum()
        probs = probs / probs.sum()
        draws = np.random.multinomial(int(n_level_total[i]), probs, size=B).astype(float)
        e1_bs[i] = draws[:, :Cr * DAYS].reshape(B, Cr, DAYS)
        e2_bs[i] = draws[:, Cr * DAYS:2 * Cr * DAYS].reshape(B, Cr, DAYS)
        n_bs[i] = e1_bs[i].sum(axis=2) + e2_bs[i].sum(axis=2) + draws[:, 2 * Cr * DAYS:]

    for i, raw in enumerate(lay.levels):
        lkey = level_slug(column, raw)
        pre = f"a5.{skey}.{lkey}"
        v1_fit = v1_by_slug[f"{skey}.{lkey}"]
        label = VD.pretty_level(column, raw)
        mask_level = keep & (codes[column] == i)
        n_all = int(mask_level.sum())
        events_all = int((mask_level & events.is_event).sum())
        info = {"column": column, "skey": skey, "lkey": lkey, "stratifier": W.STRATIFIER_DISPLAY[column],
                "level": label, "n_all": n_all, "events_all": events_all,
                "retained_n": int(n_level_total[i]), "retained_events": int(events_ret[i]),
                "suppressed": bool(suppressed[i]), "v1": v1_fit}
        put(f"{pre}.patients_n", n_all)
        put(f"{pre}.events_n", events_all)
        put(f"{pre}.retained_patients_n", int(n_level_total[i]))
        put(f"{pre}.retained_share_pct", 100.0 * int(n_level_total[i]) / n_all)
        put(f"{pre}.retained_events_n", int(events_ret[i]))
        put(f"{pre}.suppressed", bool(suppressed[i]))
        put(f"{pre}.v1_patients_n", int(v1_fit.n))
        put(f"{pre}.v1_events_n", int(v1_fit.events))
        if suppressed[i]:
            LEVEL_RESULTS.append(info)
            continue

        # point estimate: weighted and crude on the same retained patients
        e1w, e2w, weight, n_l = standardized_counts(n_pt, e1_pt, e2_pt, i)
        cif_pt, curve_pt = curves_from_counts(e1w, e2w, n_l)
        c1 = e1_pt[i].sum(axis=1)                                       # (1, D)
        c2 = e2_pt[i].sum(axis=1)
        cif_cr, curve_cr = curves_from_counts(c1, c2, n_l)
        # the weighted cell composition inside the level equals the pooled composition
        comp_w = (weight[0] * n_pt[i, 0]) / n_l[0]
        pooled = n_pt[:, 0, :].sum(axis=0)
        if not np.allclose(comp_w, pooled / pooled.sum(), atol=1e-12, rtol=0.0):
            raise R1.R1Error(f"{column} level {raw}: weighted composition differs from the pooled one")
        put(f"{pre}.max_weight", float(weight.max()))
        put(f"{pre}.min_weight", float(weight.min()))

        # bootstrap: pooled N_c, N and the weights recomputed in every replicate
        e1w_b, e2w_b, weight_b, n_l_b = standardized_counts(n_bs, e1_bs, e2_bs, i)
        empty_replicates = int((n_bs[i] == 0).any(axis=1).sum())
        put(f"{pre}.empty_cell_replicates_n", empty_replicates)
        cif_b, curve_b = curves_from_counts(e1w_b, e2w_b, n_l_b)
        c1_b = e1_bs[i].sum(axis=1)                                     # (B, D)
        c2_b = e2_bs[i].sum(axis=1)
        cif_cb, curve_cb = curves_from_counts(c1_b, c2_b, n_l_b)

        std = threshold_stats(curve_b, cif_b, curve_pt[0], cif_pt[0, -1])
        crude = threshold_stats(curve_cb, cif_cb, curve_cr[0], cif_cr[0, -1])
        info["std"], info["crude"] = std, crude
        info["empty_replicates"] = empty_replicates
        for tag, est in (("std", std), ("crude", crude)):
            put(f"{pre}.{tag}.cif30_pct", 100.0 * est["cif30"])
            put(f"{pre}.{tag}.cif30_lo_pct", 100.0 * est["cif30_lo"])
            put(f"{pre}.{tag}.cif30_hi_pct", 100.0 * est["cif30_hi"])
            for t in THRESHOLDS:
                tslug = R1.threshold_slug(t)
                put(f"{pre}.{tag}.{tslug}.day", est[t]["day"])
                put(f"{pre}.{tag}.{tslug}.lo_day", est[t]["lo"])
                put(f"{pre}.{tag}.{tslug}.hi_day", est[t]["hi"])
                put(f"{pre}.{tag}.{tslug}.stays_below", est[t]["stays_below"])
        # v1 full-cohort crude (Table 5) for the same level
        put(f"{pre}.v1.cif30_pct", 100.0 * v1_fit.cif30)
        put(f"{pre}.v1.cif30_lo_pct", 100.0 * v1_fit.boot.cif30_lo)
        put(f"{pre}.v1.cif30_hi_pct", 100.0 * v1_fit.boot.cif30_hi)
        info["v1_days"] = {}
        for t in THRESHOLDS:
            tslug = R1.threshold_slug(t)
            day = v1_fit.threshold_day(t)
            d_lo, d_hi = W.percentile_ci(v1_fit.boot_threshold_days(t).astype(float))
            stays = bool(R1.stays_below_from(v1_fit.curve, t, day))
            info["v1_days"][t] = (day, int(math.floor(d_lo)), int(math.ceil(d_hi)), stays)
            put(f"{pre}.v1.{tslug}.day", day)
            put(f"{pre}.v1.{tslug}.lo_day", int(math.floor(d_lo)))
            put(f"{pre}.v1.{tslug}.hi_day", int(math.ceil(d_hi)))
            put(f"{pre}.v1.{tslug}.stays_below", stays)
        LEVEL_RESULTS.append(info)

# overall dropped shares across the four stratifiers
put("a5.dropped.any_stratifier_patients_n", int(dropped_union.sum()))
put("a5.dropped.any_stratifier_share_pct", 100.0 * int(dropped_union.sum()) / n_pop)
put("a5.dropped.every_stratifier_patients_n", int(dropped_all.sum()))
shares = [100.0 * s["dropped_n"] / n_pop for s in STRAT_RESULTS]
put("a5.dropped.largest_share_pct", max(shares))
put("a5.dropped.smallest_share_pct", min(shares))
put("a5.empty_cell_replicates_total_n", int(sum(r.get("empty_replicates", 0) for r in LEVEL_RESULTS)))

# ---------------------------------------------------------------------------
# 6. Gradient facts: range over levels and the extreme-level contrast, crude versus standardized
# ---------------------------------------------------------------------------

for column in COLUMNS:
    skey = stratifier_slug(column)
    levels = [r for r in LEVEL_RESULTS if r["column"] == column and not r["suppressed"]]
    for t in THRESHOLDS:
        tslug = R1.threshold_slug(t)
        crude_days = [r["crude"][t]["day"] for r in levels]
        std_days = [r["std"][t]["day"] for r in levels]
        put(f"a5.spread.{skey}.{tslug}.crude_range_day", max(crude_days) - min(crude_days))
        put(f"a5.spread.{skey}.{tslug}.std_range_day", max(std_days) - min(std_days))
        first, last = levels[0], levels[-1]            # age 18-49 vs 75+, ASA 1-2 vs 3+, 1 vs 3+ levels, male vs female
        put(f"a5.contrast.{skey}.{tslug}.reference_level_slug", first["lkey"])
        put(f"a5.contrast.{skey}.{tslug}.comparison_level_slug", last["lkey"])
        for tag in ("crude", "std"):
            point = last[tag][t]["day"] - first[tag][t]["day"]
            boot = last[tag][t]["boot_days"].astype(int) - first[tag][t]["boot_days"].astype(int)
            lo, hi = W.percentile_ci(boot.astype(float))
            put(f"a5.contrast.{skey}.{tslug}.{tag}_difference_day", int(point))
            put(f"a5.contrast.{skey}.{tslug}.{tag}_difference_lo_day", int(math.floor(lo)))
            put(f"a5.contrast.{skey}.{tslug}.{tag}_difference_hi_day", int(math.ceil(hi)))
            put(f"a5.contrast.{skey}.{tslug}.{tag}_unreached_replicates_n",
                int(((last[tag][t]["boot_days"] > 30) | (first[tag][t]["boot_days"] > 30)).sum()))

# ---------------------------------------------------------------------------
# 7. Table S9
# ---------------------------------------------------------------------------


def pct_ci_text(value, lo, hi):
    return f"{100 * value:.2f} ({VD.fmt_ci(100 * lo, 100 * hi, 2)})"


def day_ci(est_t):
    return f"{est_t['day']} ({W.day_ci_text(est_t['lo'], est_t['hi'])})" if est_t["day"] <= 30 else W.NOT_REACHED


def v1_day_text(tuple_):
    day, lo, hi, _ = tuple_
    return f"{day} ({W.day_ci_text(lo, hi)})" if day <= 30 else W.NOT_REACHED


def yes_no(est_t):
    if est_t["day"] > 30:
        return W.NOT_APPLICABLE
    return "Yes" if est_t["stays_below"] else "No"


header = [
    "Stratifier", "Level", "Patients", "Composite events", "Denominator, retained patients",
    "Retained share, %", "Composite events among retained patients",
    "Crude 30-day cumulative incidence, retained patients, % (95% confidence interval)",
    "Standardized 30-day cumulative incidence, % (95% confidence interval)",
    "Crude 0.5% threshold day, retained patients (95% confidence interval)",
    "Standardized 0.5% threshold day (95% confidence interval)",
    "Crude 0.25% threshold day, retained patients (95% confidence interval)",
    "Standardized 0.25% threshold day (95% confidence interval)",
    "Crude 0.5% residual risk stays below through day 30",
    "Standardized 0.5% residual risk stays below through day 30",
    "Crude 0.25% residual risk stays below through day 30",
    "Standardized 0.25% residual risk stays below through day 30",
    "Table 5 crude 30-day cumulative incidence, full cohort, % (95% confidence interval)",
    "Table 5 crude 0.5% threshold day, full cohort (95% confidence interval)",
    "Table 5 crude 0.25% threshold day, full cohort (95% confidence interval)",
]
rows = []
for r in LEVEL_RESULTS:
    v1_fit = r["v1"]
    if r["suppressed"]:
        rows.append([r["stratifier"], r["level"], VD.fmt_count(r["n_all"]), VD.fmt_count(r["events_all"]),
                     VD.fmt_count(r["retained_n"]), f"{100 * r['retained_n'] / r['n_all']:.1f}",
                     VD.fmt_count(r["retained_events"])] + [W.SUPPRESSED_TEXT] * 13)
        continue
    c, s = r["crude"], r["std"]
    rows.append([
        r["stratifier"], r["level"], VD.fmt_count(r["n_all"]), VD.fmt_count(r["events_all"]),
        VD.fmt_count(r["retained_n"]), f"{100 * r['retained_n'] / r['n_all']:.1f}",
        VD.fmt_count(r["retained_events"]),
        pct_ci_text(c["cif30"], c["cif30_lo"], c["cif30_hi"]),
        pct_ci_text(s["cif30"], s["cif30_lo"], s["cif30_hi"]),
        day_ci(c[0.005]), day_ci(s[0.005]), day_ci(c[0.0025]), day_ci(s[0.0025]),
        yes_no(c[0.005]), yes_no(s[0.005]), yes_no(c[0.0025]), yes_no(s[0.0025]),
        pct_ci_text(v1_fit.cif30, v1_fit.boot.cif30_lo, v1_fit.boot.cif30_hi),
        v1_day_text(r["v1_days"][0.005]), v1_day_text(r["v1_days"][0.0025]),
    ])

# the 35 non-binary patients: excluded from A5 (sex is a standardizing variable in three of the four
# comparisons and they cannot form a cell); their crude estimate is withheld, 0 events of the composite
nb = v1_by_slug["sex.not-recorded"]
put("a5.sex.non-binary.patients_n", int(nb.n))
put("a5.sex.non-binary.events_n", int(nb.events))
put("a5.sex.non-binary.suppressed", bool(nb.suppressed))
nb_label = nb.stratum.level
rows.append([W.STRATIFIER_DISPLAY["MALE"], nb_label, VD.fmt_count(nb.n), VD.fmt_count(nb.events),
             W.NOT_APPLICABLE, W.NOT_APPLICABLE, W.NOT_APPLICABLE, W.SUPPRESSED_TEXT, W.NOT_APPLICABLE,
             W.SUPPRESSED_TEXT, W.NOT_APPLICABLE, W.SUPPRESSED_TEXT, W.NOT_APPLICABLE,
             W.SUPPRESSED_TEXT, W.NOT_APPLICABLE, W.SUPPRESSED_TEXT, W.NOT_APPLICABLE,
             W.SUPPRESSED_TEXT, W.SUPPRESSED_TEXT, W.SUPPRESSED_TEXT])

blank = [""] * (len(header) - 2)
share_sentences = []
for s in STRAT_RESULTS:
    pct_dropped = 100.0 * s["dropped_n"] / n_pop
    names = "; ".join(s["dropped_cells"]) if s["dropped_cells"] else "none"
    share_sentences.append(f"{W.STRATIFIER_DISPLAY[s['column']]}: {VD.fmt_count(s['dropped_n'])} patients "
                           f"({pct_dropped:.2f}%), cells dropped: {names}")
crude_vs_table5 = (
    f"Comparison with Table 5 in the main text: the crude columns are recomputed on the {VD.fmt_count(n_pop)} men and "
    f"women analysed here (the {n_nonbinary} patients recorded as non-binary are excluded), with independent bootstrap "
    f"draws, so the patient counts and, by about one day, some interval limits differ from Table 5 in the main text "
    f"while the point values agree to within rounding. The crude contrast between extreme levels should be quoted from "
    f"Table S5B (paired bootstrap in the full cohort) and the standardized contrast from this table.")
put("a5.note.crude_vs_table5", crude_vs_table5)
notes = [
    ["Note", "Standardization: each level is reweighted to the pooled distribution of the other three "
             "stratifiers (age group, ASA class, fusion levels, sex) over their cross-classified cells; "
             "the 30-day cumulative incidence and the threshold days come from the same Aalen-Johansen "
             "estimator, death as the competing event, on the weighted daily counts."] + blank,
    ["Note", f"Cell-size rule: a cell is dropped from the standard population when any level of the "
             f"stratifier being compared has fewer than {MIN_CELL} patients in it. Crude columns are recomputed on "
             f"the same retained patients as the standardized columns; the Table 5 columns use the full cohort."] + blank,
    ["Note", f"The {n_nonbinary} patients whose recorded sex is neither male nor female are excluded from this "
             f"analysis ({VD.fmt_count(n_pop)} patients analysed); they had 0 composite events."] + blank,
    ["Note", "Patients dropped by the cell-size rule, share of the " + VD.fmt_count(n_pop) + " analysed. "
             + " | ".join(share_sentences) + "."] + blank,
    ["Note", f"Intervals: percentile intervals from {B} bootstrap replicates in which each level is resampled "
             f"within its retained cells and the standard population and weights are recomputed every replicate. "
             f"A level with fewer than {VD.MIN_STRATUM_EVENTS} composite events is withheld."] + blank,
    ["Note", crude_vs_table5] + blank,
]
R1.write_csv_table(R1.TABLES_OUT / "tableS9_adjusted_strata.csv", header, rows + notes)

n_keys = R1.write_fragment(R1.FRAGMENTS_OUT / "results_a5.json", R, PREFIX)

# ---------------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------------

print(f"[a5] analysed {n_pop:,} patients; excluded {n_nonbinary} non-binary (0 events); "
      f"sanity (i) {sanity['null_cells_checked_n']} strata exact, (ii) {sanity['own_cells_checked_n']} levels, "
      f"max abs diff {sanity['own_max_abs_diff']:.2e}")
for s in STRAT_RESULTS:
    print(f"[a5] {W.STRATIFIER_DISPLAY[s['column']]}: dropped {s['dropped_n']:,} patients "
          f"({100.0 * s['dropped_n'] / n_pop:.2f}%); cells: {s['dropped_cells'] or 'none'}")
for column in COLUMNS:
    print(f"[a5] {W.STRATIFIER_DISPLAY[column]}: smallest level-by-cell count "
          f"{R['a5.' + stratifier_slug(column) + '.smallest_cell_level_n']} "
          f"({R['a5.' + stratifier_slug(column) + '.smallest_cell_label']}); cells with a level under 50: "
          f"{R['a5.' + stratifier_slug(column) + '.cells_with_a_level_below_50_n']} of {LAYOUTS[column].n_cells}")
print(f"[a5] dropped under at least one stratifier: {int(dropped_union.sum()):,} "
      f"({100.0 * dropped_union.sum() / n_pop:.2f}%); under all four: {int(dropped_all.sum()):,}")
for r in LEVEL_RESULTS:
    if r["suppressed"]:
        print(f"[a5] {r['stratifier']} / {r['level']}: suppressed")
        continue
    c, s = r["crude"], r["std"]
    print(f"[a5] {r['stratifier']:<26s} {r['level']:<18s} n={r['n_all']:>6,} ret={r['retained_n']:>6,} "
          f"CIF30 crude {100 * c['cif30']:.3f} std {100 * s['cif30']:.3f} | "
          f"0.5%: v1 {r['v1_days'][0.005][0]} crude {c[0.005]['day']} std {s[0.005]['day']} "
          f"({s[0.005]['lo']}-{s[0.005]['hi']}) | "
          f"0.25%: v1 {r['v1_days'][0.0025][0]} crude {c[0.0025]['day']} std {s[0.0025]['day']} "
          f"({s[0.0025]['lo']}-{s[0.0025]['hi']}) | empty reps {r['empty_replicates']} "
          f"maxw {R[f'a5.{r['skey']}.{r['lkey']}.max_weight']:.2f}")
print(f"[a5] wrote {n_keys} keys to results_a5.json")
