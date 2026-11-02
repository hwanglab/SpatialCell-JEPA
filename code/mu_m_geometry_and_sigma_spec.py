#!/usr/bin/env python3
"""mu_m_geometry_and_sigma_spec.py — move the cos(mu,m) and sigma^2 numbers onto the SPEC predictor.

WHY THIS EXISTS
[user 2026-08-24] asked (a) why the cos(mu,m) result was never in the paper and (b) to bring the
sigma^2 analysis into the experimental section. Both existing numbers come from
`results/reports/20260721_192542_t2_mu_sigma_validation/` which ran on **predictor_out_fine112_m0**
-- an OLDER predictor (fine fov 112, pre-corpus31, pre-d2/per-element-PE) on **3 slides**. Table 1 of
the paper is the **fov56 d2_perel** spec predictor on 12 slides. Quoting the old numbers beside
Table 1 would repeat exactly the mistake caught on 2026-08-24: the fov56 per-class analysis had been
computed on the wrong predictor and it went unnoticed because `m` is predictor-INDEPENDENT, so the
`m` column matched while `mu` silently did not.

⚠ NOT recomputed here, because they ALREADY use the spec predictor (verified in their PROVENANCE):
  - the morphology-ambiguity strata  `20260818_195021_t2_positive_control_margin_nystroem`
  - the reconstruction R^2           `20260818_164826_t2_within_slide_r2`
Only cos(mu,m) and the sigma^2 battery were stale.

WHAT IT COMPUTES  (parses cached artifacts; NO model forward, NO GPU)
  A RECONSTRUCTION  cos(mu,m) uncentered AND slide-centered; R^2 total and within-slide;
                    per-cell MSE vs two baselines (36-NN mean of m, global mean).
  B SIGMA           non-degeneracy vs the logvar clamp, calibration E[r^2]/E[sigma^2] and
                    Spearman(sigma, r^2), niche-mixing Spearman(sigma, neighbour-type entropy),
                    confounds (local density, ||m||, slide), sigma by class.

⚠⚠ THE CENTERING CAVEAT, and it is the reason cos alone overstates the result.
  All embeddings from one encoder occupy a narrow cone, so an UNCENTERED cosine is inflated by the
  shared mean direction. R^2 is computed after centering. The two therefore answer different
  questions -- "does mu point the right way" vs "does mu capture the cell-to-cell deviation" -- and
  the paper must report BOTH or neither. This script emits both from the same cells so they cannot
  drift apart. Slide-centered uses the slide's own mean of `m` over the scored cells; that is a
  DESCRIPTIVE statistic, not a fitted quantity, so it is not a leak, but it is stated rather than
  implied.

⚠ `sigma` here is mean_d sqrt(s2_d), matching `sbar` in the 2026-07-21 battery (scale ~0.12), NOT
  the variance. Stated because "sigma^2" is used loosely in the notes.
⚠ The 36-NN mean baseline uses ALL cells as neighbours (labelled or not), which is the rule
  `p2_typing_and_slideid.build()` uses for its `smooth` arm; the neighbour-TYPE entropy necessarily
  uses labelled cells only. Two different neighbour sets, both stated.

READ-ONLY on all inputs.
"""
import argparse, glob, json, os, hashlib, datetime
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

CLASSES = ["Immune", "Stromal", "Tumor"]
MPP = 0.2522
LOGVAR_CLAMP = (-7.0, 7.0)          # scripts/model/predictor_mssc.py:37
S2_FLOOR, S2_CEIL = float(np.exp(LOGVAR_CLAMP[0])), float(np.exp(LOGVAR_CLAMP[1]))
# cross-check target from the run that already used this predictor (different cell subsample =>
# expect close, not exact). Reported loudly, never asserted.
R2_EXPECTED = {"total": 0.5372, "within_slide": 0.5259,
               "src": "results/reports/20260818_164826_t2_within_slide_r2/results.json"}


def md5(p, cap=32 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(1 << 20)):
            h.update(b)
            if f.tell() > cap:
                break
    return h.hexdigest()


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache_fm_hd/hoptimus0")
    ap.add_argument("--mu-dir", required=True)
    ap.add_argument("--labels", default="results/eval/gt_labels_hd12")
    ap.add_argument("--fov", default="fov56")
    ap.add_argument("--k", type=int, default=36)
    ap.add_argument("--per-slide", type=int, default=30000,
                    help="cells scored per slide (0 = all labelled qc_pass). 30000 matches the "
                         "2026-07-21 battery this supersedes.")
    ap.add_argument("--slides", default="", help="comma-separated subset; default = all 12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    explicit = [s.strip() for s in a.slides.split(",") if s.strip()]
    slides = explicit or sorted(os.path.basename(p).replace(".parquet", "")
                                for p in glob.glob(f"{a.labels}/*.parquet"))
    # The eval cohort is the 12-slide Visium HD set (see the paper's slide-composition appendix)
    # Sources. Globbing a directory is a level-5 source, so the canonical COUNT is asserted rather
    # than trusted: a silently-11 glob would produce a plausible, wrong summary.
    if not explicit:
        assert len(slides) == 12, (f"expected the canonical 12 HD eval slides, glob found "
                                   f"{len(slides)}: {slides}")
    per_slide, rows, inputs = [], [], {}

    for s in slides:
        cpath, mpath, lpath = (f"{a.cache}/{s}/{a.fov}/cells.npz",
                              f"{a.mu_dir}/{s}.npz", f"{a.labels}/{s}.parquet")
        for p in (cpath, mpath, lpath):
            assert os.path.exists(p), f"missing {p}"
        z = np.load(cpath)
        m_all = z["m_cell"].astype(np.float32)
        xy = z["centroid_xy"].astype(np.float32) * MPP
        g = np.load(mpath)
        assert g["mu"].shape == m_all.shape, f"{s}: mu {g['mu'].shape} vs m {m_all.shape}"
        assert g["s2"].shape == m_all.shape, f"{s}: s2 {g['s2'].shape} vs m {m_all.shape}"
        lab = pd.read_parquet(lpath)
        keep = np.where(lab.qc_pass.fillna(False).values & lab.L1.isin(CLASSES).values)[0]
        if a.per_slide and len(keep) > a.per_slide:
            keep = rng.choice(keep, a.per_slide, replace=False)
        keep.sort()
        ri = lab.row_idx.values[keep]
        y = lab.L1.values[keep]

        m = m_all[ri]
        mu = g["mu"][ri].astype(np.float32)
        s2 = g["s2"][ri].astype(np.float32)
        inputs[s] = {"cache_md5": md5(cpath), "mu_md5": md5(mpath),
                     "n_cells_total": int(len(m_all)), "n_scored": int(len(ri))}

        # ---- A RECONSTRUCTION -----------------------------------------------------------
        num = (mu * m).sum(1)
        cos_un = num / (np.linalg.norm(mu, axis=1) * np.linalg.norm(m, axis=1) + 1e-8)
        mbar = m.mean(0, keepdims=True)                     # slide mean of m over scored cells
        mc, muc = m - mbar, mu - mbar
        cos_ct = ((muc * mc).sum(1) /
                  (np.linalg.norm(muc, axis=1) * np.linalg.norm(mc, axis=1) + 1e-8))

        r2_cell = ((m - mu) ** 2).mean(1)                   # per-cell MSE, ours
        ss_res = float(((m - mu) ** 2).sum())
        ss_tot_within = float(((m - mbar) ** 2).sum())

        # 36-NN mean of m over ALL cells (the `smooth` rule), then its MSE
        _, nn = cKDTree(xy).query(xy[ri], k=a.k + 2, workers=-1)
        sm = np.empty_like(m)
        for i, (row, self_i) in enumerate(zip(nn, ri)):
            sm[i] = m_all[row[row != self_i][:a.k]].mean(0)
        mse_nbr = ((m - sm) ** 2).mean(1)
        mse_slidemean = ((m - mbar) ** 2).mean(1)   # SLIDE mean, not a global mean
        kdist = np.asarray([d[-1] for d in _]) if False else _[:, -1]

        # ---- B SIGMA ---------------------------------------------------------------------
        sig = np.sqrt(s2)
        sbar = sig.mean(1)
        r2_dim_mean, s2_dim_mean = float(((m - mu) ** 2).mean()), float(s2.mean())
        frac_floor = float((s2 <= S2_FLOOR * 1.01).mean())
        frac_ceil = float((s2 >= S2_CEIL * 0.99).mean())

        # neighbour-TYPE entropy over LABELLED cells only
        tl = cKDTree(xy[ri])
        _, nnl = tl.query(xy[ri], k=a.k + 1, workers=-1)
        codes = pd.Categorical(y, categories=CLASSES).codes
        ent = np.zeros(len(ri), np.float32)
        for i in range(len(ri)):
            idx = nnl[i][nnl[i] != i][:a.k]
            c = np.bincount(codes[idx], minlength=len(CLASSES)).astype(float)
            p = c / max(c.sum(), 1)
            p = p[p > 0]
            ent[i] = float(-(p * np.log(p)).sum())

        per_slide.append(dict(
            slide=s, n=int(len(ri)),
            cos_uncentered_median=float(np.median(cos_un)),
            cos_slidecentered_median=float(np.median(cos_ct)),
            mse_ours_median=float(np.median(r2_cell)),
            mse_nbr36_median=float(np.median(mse_nbr)),
            mse_slidemean_median=float(np.median(mse_slidemean)),
            r2_within_slide=float(1.0 - ss_res / ss_tot_within),
            sbar_median=float(np.median(sbar)),
            sbar_p05=float(np.percentile(sbar, 5)), sbar_p95=float(np.percentile(sbar, 95)),
            frac_s2_at_floor=frac_floor, frac_s2_at_ceil=frac_ceil,
            calib_ratio_r2_over_s2=r2_dim_mean / s2_dim_mean,
            spearman_sigma_vs_r2=sp(sbar, r2_cell),
            spearman_sigma_vs_nbr_entropy=sp(sbar, ent),
            spearman_sigma_vs_density=sp(sbar, kdist),
            spearman_sigma_vs_mnorm=sp(sbar, np.linalg.norm(m, axis=1)),
            **{f"sbar_{c}": float(sbar[y == c].mean()) if (y == c).any() else float("nan")
               for c in CLASSES}))
        rows.append((m, mu, mbar, y, sbar))
        print(f"  {s:22s} n={len(ri):6d} cos_un={per_slide[-1]['cos_uncentered_median']:.4f} "
              f"cos_ct={per_slide[-1]['cos_slidecentered_median']:.4f} "
              f"R2w={per_slide[-1]['r2_within_slide']:.4f} "
              f"sbar={per_slide[-1]['sbar_median']:.4f}", flush=True)
        del m_all, z, g

    ps = pd.DataFrame(per_slide)

    # ---- pooled R^2: total uses a GLOBAL mean, within-slide uses each slide's own -----------
    allm = np.concatenate([r[0] for r in rows])
    allmu = np.concatenate([r[1] for r in rows])
    gbar = allm.mean(0, keepdims=True)
    r2_total = float(1.0 - ((allm - allmu) ** 2).sum() / ((allm - gbar) ** 2).sum())
    ss_res_w = sum(float(((r[0] - r[1]) ** 2).sum()) for r in rows)
    ss_tot_w = sum(float(((r[0] - r[2]) ** 2).sum()) for r in rows)
    r2_within = float(1.0 - ss_res_w / ss_tot_w)
    ally = np.concatenate([r[3] for r in rows]); allsb = np.concatenate([r[4] for r in rows])

    summary = {
        "maturity": f"[FINAL n={len(slides)}/{len(slides)} slides]",
        "n_cells_scored": int(len(allm)),
        "cos_uncentered_median": float(np.median(np.concatenate(
            [[p["cos_uncentered_median"]] for p in per_slide]))),
        "cos_uncentered_median_of_slide_medians": float(ps.cos_uncentered_median.median()),
        "cos_slidecentered_median_of_slide_medians": float(ps.cos_slidecentered_median.median()),
        "r2_total_globalmean": r2_total,
        "r2_within_slide": r2_within,
        "r2_crosscheck_expected": R2_EXPECTED,
        "mse_ours_median_of_slides": float(ps.mse_ours_median.median()),
        "mse_nbr36_median_of_slides": float(ps.mse_nbr36_median.median()),
        "mse_slidemean_median_of_slides": float(ps.mse_slidemean_median.median()),
        "sbar_median_of_slides": float(ps.sbar_median.median()),
        "sbar_slide_spread": [float(ps.sbar_median.min()), float(ps.sbar_median.max())],
        "frac_s2_at_floor_max": float(ps.frac_s2_at_floor.max()),
        "frac_s2_at_ceil_max": float(ps.frac_s2_at_ceil.max()),
        "calib_ratio_median": float(ps.calib_ratio_r2_over_s2.median()),
        "spearman_sigma_vs_r2_median": float(ps.spearman_sigma_vs_r2.median()),
        "spearman_sigma_vs_nbr_entropy_median": float(ps.spearman_sigma_vs_nbr_entropy.median()),
        "spearman_sigma_vs_nbr_entropy_slidewise_sign_consistent":
            bool((ps.spearman_sigma_vs_nbr_entropy > 0).all() or
                 (ps.spearman_sigma_vs_nbr_entropy < 0).all()),
        "spearman_sigma_vs_density_median": float(ps.spearman_sigma_vs_density.median()),
        "spearman_sigma_vs_mnorm_median": float(ps.spearman_sigma_vs_mnorm.median()),
        "sbar_by_class_pooled": {c: float(allsb[ally == c].mean()) for c in CLASSES},
        "centering_basis": "slide mean of m over the scored cells (descriptive, not fitted)",
        "sigma_definition": "mean over the 1536 dims of sqrt(s2); NOT the variance",
        "nbr_sets": {"mse_nbr36": "36-NN over ALL cells (the `smooth` rule)",
                     "type_entropy": "36-NN over LABELLED cells only"},
    }

    ps.to_csv(f"{a.out}/per_slide.tsv", sep="\t", index=False)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{a.out}/PROVENANCE.json", "w") as f:
        json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "script": os.path.abspath(__file__),
                   "maturity": summary["maturity"],
                   "predictor": os.path.abspath(a.mu_dir),
                   "supersedes": "results/reports/20260721_192542_t2_mu_sigma_validation "
                                 "(predictor_out_fine112_m0, 3 slides) -- WRONG predictor for Table 1",
                   "params": {k: getattr(a, k) for k in
                              ("cache", "mu_dir", "labels", "fov", "k", "per_slide", "seed")},
                   "inputs": inputs,
                   "logvar_clamp": LOGVAR_CLAMP,
                   "caveats": [
                       "UNCENTERED cosine is inflated by the encoder's shared mean direction; "
                       "report it only alongside the centered cosine and R^2.",
                       "R^2 cross-check is against a run with a different cell subsample => close, "
                       "not exact; it is printed, never asserted.",
                   ]}, f, indent=2)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    d_t = abs(r2_total - R2_EXPECTED["total"])
    d_w = abs(r2_within - R2_EXPECTED["within_slide"])
    flag = "OK" if max(d_t, d_w) < 0.02 else "⚠ DIVERGES"
    print(f"\n[R2 cross-check] total {r2_total:.4f} vs {R2_EXPECTED['total']} (d={d_t:.4f}) · "
          f"within {r2_within:.4f} vs {R2_EXPECTED['within_slide']} (d={d_w:.4f}) => {flag}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
