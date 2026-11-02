#!/usr/bin/env python3
"""P16 — Does mu behave differently when the neighbourhood is type-HOMOGENEOUS vs HETEROGENEOUS?

★ THE QUESTION [user 2026-08-25]. Section 3.3 of the workshop paper is the longest subsection and
delivers no key message: it says mu != m, but not complementary, except conditionally, and here is a
sigma we claim nothing from. The user's original intent was to show that mu carries information
COMPLEMENTARY to m and to CHARACTERISE mu. This script supplies the missing measurement:

    compare mu in regions where the nearest neighbours are a DIFFERENT cell type from the target
    against regions where they are the SAME type.

[user] "based on our design 1) and 2) should be different if not I think there would some error we
did not notice."

★★ WHAT IS MEASURED PER STRATUM (the user's five items)
  (1)/(2) the strata themselves          -> own-type fraction over the k nearest neighbours
  (3)     sigma                          -> mean over 1536 dims of s2, sqrt
  (4)     cell typing                    -> macro-F1 per arm + PER-CLASS recall/precision
  (5)     similarities                   -> cos(m, mu) AND cos(m, neighbour m)   [user: "both"]

★★★ THE LEAKAGE PROBLEM, AND THE FIX (this is why there are two stratifier families)
Defining "the neighbours are the same type as the target" from GT labels reads HELD-OUT LABELS. In
the homogeneous stratum the target's label IS the neighbour majority BY CONSTRUCTION, so a
neighbour-derived mu scores well there because the stratum definition contains the answer. That is
the failure mode `feedback_holdout_contamination_no_tuning_on_validation` and
`feedback_check_control_not_vacuous` both describe. Note Table 3's margin stratifier is explicitly
label-free (ms.md: "without using any held-out label") -- so a GT-based stratifier would NOT be
comparable to it.

    f8_pred / f36_pred  PRIMARY     neighbour labels PREDICTED by the m-only probe fitted on the 11
                                    TRAINING slides and applied to the held-out slide's FULL labelled
                                    feature set. Never sees a held-out label. Comparable to Table 3.
    f8_gt / f36_gt      DIAGNOSTIC  neighbour labels from GT. Answers "does the model respond to true
                                    neighbourhood composition at all" -- a mechanism/bug check.
                                    ⚠ MUST NOT be presented as a performance result. [user 2026-08-25
                                    asked for it to be run: "run at all".]

★ NEIGHBOUR SET [user asked: K=36 or effective NN 8-12?]
  k=8 PRIMARY.  Section 3.2 measures effective K = 7.35 and the most-attended neighbour inside the
  nearest-8 for 95% of cells, so defining homogeneity over 36 dilutes the contrast with 28 cells the
  model does not read.  k=36 is carried as ROBUSTNESS (it is the actual context set size, Kc=36).
  ⚠ Neighbours are ranked by DISTANCE, never by attention: selecting by the model's own attention
    would make the stratifier a function of the model under test.

★ BINNING [user asked: all neighbours, or most?]
  Own-type fraction over k=8 takes only 9 distinct values, so quantile bins collapse. Binning is
  therefore on the NATURAL LEVELS, plus an explicit coarse 3-bin view:
      f == 0        "all neighbours a different type"   <- the user's case (1)
      0 < f < 1     mixed
      f == 1        "all neighbours the same type"      <- the user's case (2)
  ⚠ f==0 is ~4% of cells (measured 12/12, gate report 20260825_gate_neighborhood_homogeneity)
    => ~480 cells per fold at --per-slide 12000. n and MDE80 are reported per stratum; a stratum
    thinner than --min-cells is dropped, loudly.

★★ PRE-REGISTERED, fixed before any number exists:
  GATE C   rho(f8_pred, margin) per fold, REPORTED FIRST. Table 3 already stratifies on the m-probe
           margin. If |rho| is large this experiment RE-DERIVES Table 3 through a different
           stratifier and must NOT be presented as a second, independent result.
  GATE R   regression: arm `m` at stratifier=ALL must reproduce the standing value EXACTLY --
           0.6159 (logistic) / 0.6174 (nystroem). A run whose `m` moves has unreadable strata.
  PRIMARY  delta = macro-F1(mu) - macro-F1(m), paired over the 12 LOPO folds, in f8_pred == 0 vs == 1.
  SHAPE    the user's own prediction: the two extremes should DIFFER. Flat on the ABSOLUTE mu
           numbers => a real bug (mu is not reading the neighbourhood). Flat on DELTA => not a bug;
           it means m and mu degrade together and there is no complementarity to characterise.
  CONFOUND own-type fraction is class-confounded (Immune mean f8 0.560 vs Tumor 0.759; eta^2 = 0.064
           measured 12/12). Every primary contrast is ALSO reported WITHIN CLASS.
           ⚠ macro-F1 is undefined within a single class, so the within-class readout is per-class
             RECALL/PRECISION, plus a class-reweighted macro-F1 for the headline.
  ⚠⚠ LABEL-CEILING CONFOUND -- stated BEFORE the run, per PROJECT_NOTES.md:464 (2026-08-19), which
     pre-registered it for exactly this stratifier: the GT labels are RNA-derived and spatially
     smoothed, so THE LABEL CEILING IS ITSELF PROBABLY STRATUM-DEPENDENT -- labels are likely MORE
     accurate in homogeneous (high-f8) neighbourhoods. Consequences for reading the primary:
       * "context helps in MIXED regions" and "labels are WORSE in mixed regions" predict OPPOSITE
         signs, so that direction is separable by this design.
       * "context helps in HOMOGENEOUS regions" is PARTLY CONFOUNDED with the labels simply being
         easier there, and CANNOT be disentangled by this run. If the win lands at f8==1, it must be
         reported with this confound attached, not as a clean neighbourhood effect.
     ⚠ This design does not measure the label ceiling per stratum; nothing here estimates it.
  ⚠ SMOKE PRIOR, recorded before the full run (gate, [SMOKE n=4/12]): sigma is expected to be another
    null -- sign inconsistent across slides, and consistent with the 2026-08-24 spec recompute where
    Spearman(sigma, niche entropy) = 0.106 with 3/12 folds negative, recorded as DIED. cos(m, nbr m)
    is expected to be the one quantity moving in all three classes.

⚠ C is FIXED at 0.01, never best_C. Quantile/level bins are cut WITHIN the held-out slide, because
  mean f8 ranges 0.582-0.788 across slides and global bins would shift meaning between folds.
⚠ Own-type fraction is computed on the FULL labelled cell set of each slide and only THEN attached
  to the subsample. Recomputing it inside a 12k subsample would inflate the nearest-8 radius ~4.7x.
⚠ Fraction of segmented cells carrying a usable label ranges 56.6%-97.3% across slides, so the
  "8 nearest LABELLED neighbours" span a different physical radius per slide. The per-slide median
  nearest-8 radius is reported so this is visible rather than hidden.

Forked from scripts/eval/positive_control_margin_strata.py (2026-08-18), which produced Table 3.
The probe, folds, cell count, C, seed and class list are unchanged so the two are comparable.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))

CLASSES = ["Immune", "Stromal", "Tumor"]
STANDING_M = {"logistic": 0.6159, "nystroem": 0.6174}   # GATE R, verified 2026-08-25


def _median_gamma(X, seed, n=2000):
    """RBF gamma = 1 / median pairwise squared distance, TRAIN subsample only. Verbatim from the
    parent script so the two harnesses stay comparable."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    S = X[idx].astype(np.float32)
    sq = (S * S).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (S @ S.T), 0.0)
    med = float(np.median(d2[np.triu_indices_from(d2, 1)]))
    return 1.0 / med if med > 0 else 1.0


def _fit(Xtr, ytr, C, estimator, seed):
    """Returns (predict_fn, ) with the scaler/kernel captured, so the SAME fitted probe can later be
    applied to a second, much larger matrix (the full labelled slide) in chunks."""
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    ny = None
    if estimator == "nystroem":
        from sklearn.kernel_approximation import Nystroem
        ny = Nystroem(kernel="rbf", gamma=_median_gamma(Ztr, seed),
                      n_components=min(1024, len(Ztr) - 1), random_state=seed).fit(Ztr)
        Ztr = ny.transform(Ztr)
    clf = LogisticRegression(C=C, max_iter=300, class_weight="balanced", random_state=seed)
    clf.fit(Ztr, ytr)

    def predict(X, proba=False, chunk=50000):
        out = []
        for i in range(0, len(X), chunk):
            Z = sc.transform(X[i:i + chunk])
            if ny is not None:
                Z = ny.transform(Z)
            out.append(clf.predict_proba(Z) if proba else clf.predict(Z))
        return np.concatenate(out)
    return predict, clf.classes_


def mde80(sd, n):
    if not np.isfinite(sd) or sd == 0 or n < 2:
        return 0.0
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)


def _levels(v, nq):
    """Bin edges. Own-type fraction over k=8 has only 9 distinct values, so quantiles collapse to
    duplicate edges and np.digitize returns degenerate bins. Use the natural levels when the
    variable is coarsely discrete, quantiles otherwise."""
    u = np.unique(v)
    if len(u) <= 12:
        return {f"{x:.4g}": (v == x) for x in u}
    qs = np.quantile(v, np.linspace(0, 1, nq + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    b = np.clip(np.digitize(v, qs[1:-1]), 0, nq - 1)
    return {f"q{i}": (b == i) for i in range(nq)}


def _coarse(v):
    """The user's three cases, verbatim."""
    return {"all_diff(f=0)": v == 0.0,
            "mixed(0<f<1)": (v > 0.0) & (v < 1.0),
            "all_same(f=1)": v == 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache_fm_hd/hoptimus0")
    ap.add_argument("--mu-dir", default="results/eval/predout_corpus31_fov56_d2_perel")
    ap.add_argument("--labels", default="results/eval/gt_labels_hd12")
    ap.add_argument("--fov", type=int, default=56)
    ap.add_argument("--per-slide", type=int, default=12000)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--nq", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 36],
                    help="neighbour-set sizes; the FIRST is primary")
    ap.add_argument("--nbr-k", type=int, default=36,
                    help="k for the TRAINING-FREE neighbour-average arms `nbr` and `m+nbr`. "
                         "Default 36 = the predictor's own K_c (ms.md:687), so the average reads the "
                         "same number of neighbours the predictor does. [user 2026-08-26] asked for "
                         "'the cell typing on the average embedding ... compare our model to the "
                         "average neighboring cells' embeddings'. ⚠ Neighbours are taken over ALL "
                         "segmented cells, NOT only the labelled/QC-pass ones — [user 2026-08-26] "
                         "chose option (a), because the predictor's context is not label-filtered. "
                         "This deliberately differs from the kNN behind the stratifier and behind "
                         "cos_m_nbr, which run over labelled cells only.")
    ap.add_argument("--niche-dir", default=None,
                    help="★ T4 (2026-08-30). Dir of Niche-JEPA outputs from apply_niche_jepa.py, which "
                         "writes z/mu/s2 into ONE npz -- so this normally points at the SAME dir as "
                         "--mu-dir and only the `z` key is new. Adds arms `z` and `m+z`.\n"
                         "⚠⚠ DEFAULT OFF AND THAT IS LOAD-BEARING: with the flag absent, `z` is never "
                         "loaded, ARMS is untouched, and every number this script produces is "
                         "bit-identical to the pre-2026-08-30 path -- so GATE R and the published "
                         "08-25 homogeneity result stay undisturbed. Same discipline as the "
                         "p2_typing_and_slideid.py --niche-dir patch.\n"
                         "⚠ `z` is a PURE NEIGHBOURHOOD CODE: unlike mu it is never fitted to the "
                         "target's own m, so the f*_gt circularity described in the module docstring "
                         "is STRONGER for z than for mu. f*_gt stays DIAGNOSTIC; f*_pred carries any "
                         "claim. The contrast that actually discriminates is `z-nbr` / "
                         "`(m+z)-(m+nbr)`: if z tracks the training-free average stratum for stratum, "
                         "the bottleneck reimplemented an arithmetic mean.")
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", default="logistic", choices=["logistic", "nystroem"])
    ap.add_argument("--skip-gate", action="store_true",
                    help="do not abort when GATE R fails (for deliberate off-spec runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    k_pri = a.ks[0]

    slides = sorted(os.path.basename(p)[:-4] for p in os.listdir(a.mu_dir) if p.endswith(".npz"))
    # ★ scope CHECKED against the canonical artifact rather than trusted from the glob
    #   (feedback_lookup_scope_dont_derive). Cannot change any number -- it only refuses to run on a
    #   partial predout dir, which previously would have silently produced a <12-slide result.
    n_canon = len([p for p in os.listdir(a.labels) if p.endswith(".parquet")])
    assert len(slides) == n_canon, (
        f"--mu-dir has {len(slides)} npz but {a.labels} defines {n_canon} slides. Refusing to run on "
        f"a partial set; a short slide list silently changes every pooled number.")
    print(f"slides: {len(slides)} | estimator={a.estimator} | ks={a.ks} | "
          f"niche={a.niche_dir or 'OFF'}", flush=True)

    # ── load; own-type fraction computed on the FULL labelled set, THEN subsampled ───────────────
    M, MU, S2, Y, SL, COSMMU, COSMNBR, NBR = [], [], [], [], [], [], [], []
    Z = [] if a.niche_dir else None
    FR = {k: [] for k in a.ks}
    FULL = {}                      # per slide: full labelled m + xy + row order, for f*_pred
    radii = {}
    for si, s in enumerate(slides):
        z = np.load(os.path.join(a.cache, s, f"fov{a.fov}", "cells.npz"))
        m_all, xy_all = z["m_cell"], z["centroid_xy"]
        p = np.load(os.path.join(a.mu_dir, f"{s}.npz"))
        mu_all, s2_all = p["mu"], p["s2"]
        assert len(m_all) == len(mu_all), f"{s}: m {len(m_all)} vs mu {len(mu_all)}"

        lab = pd.read_parquet(os.path.join(a.labels, f"{s}.parquet"))
        lab = lab[lab.qc_pass & lab.L1.isin(CLASSES)]
        idx_full = lab.row_idx.to_numpy()
        y_full = pd.Categorical(lab.L1, categories=CLASSES).codes.astype(np.int8)
        xy_full = xy_all[idx_full]

        tree = cKDTree(xy_full)
        frac_full, rad, nn_full = {}, {}, {}
        for k in a.ks:
            d, nnk = tree.query(xy_full, k=k + 1)
            nn_full[k] = nnk[:, 1:].astype(np.int32)          # int32: int64 costs 2x for nothing
            rad[k] = float(np.median(d[:, -1]))
            frac_full[k] = (y_full[nn_full[k]] == y_full[:, None]).mean(1).astype(np.float32)
        radii[s] = rad

        pick = np.arange(len(idx_full))
        if len(pick) > a.per_slide:
            pick = np.sort(rng.choice(len(pick), a.per_slide, replace=False))
        idx = idx_full[pick]

        m_s = m_all[idx].astype(np.float32)
        mu_s = mu_all[idx].astype(np.float32)
        nrm = lambda A: A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
        # (5) cos(m, mu) and cos(m, neighbour m) -- [user 2026-08-25: "both"]
        mn = m_all[idx_full[nn_full[k_pri][pick]]].astype(np.float32)
        COSMMU.append((nrm(m_s) * nrm(mu_s)).sum(1))
        COSMNBR.append((nrm(mn) * nrm(m_s)[:, None, :]).sum(-1).mean(1))
        del mn

        # ── TRAINING-FREE neighbour-average arm [user 2026-08-26, option (a)] ────────────────────
        # The arithmetic mean of the --nbr-k nearest neighbours' frozen embeddings. This is the
        # `smooth` arm ms.md:803 DEFINES and never reports, and it is the baseline PROJECT_NOTES:174
        # says "matched" the learned predictor on every well-powered axis -- a claim that was never
        # tested on the one axis where mu is significant, because no run ever scored both arms.
        # ⚠ Neighbours come from a tree over ALL segmented cells (xy_all), not the labelled subset,
        # so this reads the same population the predictor's context does. Self is excluded
        # explicitly by index rather than by assuming it sorts first, since duplicate centroids
        # would break that assumption silently.
        tree_all = cKDTree(xy_all)
        nn_all = tree_all.query(xy_all[idx], k=a.nbr_k + 1)[1].astype(np.int64)
        is_self = nn_all == idx[:, None]
        keep = np.ones_like(nn_all, dtype=bool)
        has_self = is_self.any(1)
        keep[np.flatnonzero(has_self), np.argmax(is_self, axis=1)[has_self]] = False
        keep[~has_self, -1] = False                      # no self found => drop the farthest
        nn_all = nn_all[keep].reshape(len(idx), a.nbr_k)
        # chunked so the gather peaks at ~0.4 GB rather than n x k x D (2.6 GB at 12k x 36 x 1536)
        nbr_s = np.empty((len(idx), m_all.shape[1]), np.float32)
        for st in range(0, len(idx), 2000):
            sl = slice(st, min(st + 2000, len(idx)))
            nbr_s[sl] = m_all[nn_all[sl]].astype(np.float32).mean(1)
        NBR.append(nbr_s)
        del tree_all, nn_all, is_self, keep

        # ── T4 niche latent, row-aligned to the SAME `idx` as m/mu ───────────────────────────────
        if Z is not None:
            pz = np.load(os.path.join(a.niche_dir, f"{s}.npz"))
            assert "z" in pz.files, (
                f"{a.niche_dir}/{s}.npz has {pz.files}; no `z` key. --niche-dir must point at an "
                f"apply_niche_jepa.py output, not a v1 predout.")
            z_all = pz["z"]
            assert len(z_all) == len(m_all), \
                f"{s}: z has {len(z_all)} rows vs m {len(m_all)} -- NOT row-aligned to the cache"
            Z.append(z_all[idx].astype(np.float32))
            pz.close()

        M.append(m_s); MU.append(mu_s)
        S2.append(s2_all[idx].astype(np.float32).mean(1))
        Y.append(lab.L1.to_numpy()[pick])
        SL.append(np.full(len(idx), si))
        for k in a.ks:
            FR[k].append(frac_full[k][pick])
        # For the label-free stratifier we must predict neighbour types on the slide's FULL labelled
        # set. ⚠ Holding every slide's full m here costs ~15 GB, so only the small arrays are kept
        # and `m` is reloaded for the ONE held-out slide inside the fold loop.
        FULL[s] = dict(idx=idx_full, nn=nn_full, pick=pick, y=y_full)
        del m_all, mu_all, s2_all, xy_all, z, p, tree
        print(f"  [{si+1}/{len(slides)}] {s}: {len(idx_full):,} labelled -> {len(idx):,} sampled "
              f"| median nn{k_pri} radius {rad[k_pri]:.1f} um", flush=True)

    M = np.concatenate(M); MU = np.concatenate(MU); S2 = np.concatenate(S2); NBR = np.concatenate(NBR)
    Y = np.concatenate(Y); SL = np.concatenate(SL)
    COSMMU = np.concatenate(COSMMU); COSMNBR = np.concatenate(COSMNBR)
    FR = {k: np.concatenate(v) for k, v in FR.items()}
    SIG = np.sqrt(S2)
    print(f"pooled: {len(Y):,} cells", flush=True)

    # ⚠ m / mu / m+mu are UNCHANGED and must stay first: GATE R re-scores `m` against the standing
    # value and is the regression proof that adding the two nbr arms moved nothing. Cell selection
    # happens before this point and consumes no extra RNG, so the scored cells are bit-identical.
    ARMS = {"m": lambda i: M[i], "mu": lambda i: MU[i],
            "m+mu": lambda i: np.hstack([M[i], MU[i]]),
            "nbr": lambda i: NBR[i],
            "m+nbr": lambda i: np.hstack([M[i], NBR[i]])}
    if Z is not None:
        Z = np.concatenate(Z)
        assert len(Z) == len(Y), f"z has {len(Z)} pooled rows vs {len(Y)} cells"
        print(f"niche z: {Z.shape} from {a.niche_dir}", flush=True)
        # ⚠ APPENDED, never inserted before `m`: GATE R must keep re-scoring `m` on bit-identical
        #   cells, which is the proof that adding these arms moved nothing.
        ARMS["z"] = lambda i: Z[i]
        ARMS["m+z"] = lambda i: np.hstack([M[i], Z[i]])

    rows, gatec, gater = [], [], []
    for si, held in enumerate(slides):
        te = SL == si
        tr = SL != si
        ytr, yte = Y[tr], Y[te]

        # m-only probe fitted on the 11 TRAINING slides (the Table 3 stratifier, unchanged)
        pred_m, _ = _fit(M[tr], ytr, a.C, "logistic", a.seed)
        proba_m = pred_m(M[te], proba=True)
        ps = np.sort(proba_m, axis=1)
        margin = ps[:, -1] - ps[:, -2]

        # label-free own-type fraction: SAME probe, applied to the held-out slide's FULL labelled set
        f_pred = {}
        m_held = np.load(os.path.join(a.cache, held, f"fov{a.fov}", "cells.npz")
                         )["m_cell"][FULL[held]["idx"]].astype(np.float32)
        yhat_code = pd.Categorical(pred_m(m_held), categories=CLASSES).codes.astype(np.int8)
        del m_held
        for k in a.ks:
            nnk = FULL[held]["nn"][k]
            f_pred[k] = (yhat_code[nnk] == yhat_code[:, None]).mean(1).astype(
                np.float32)[FULL[held]["pick"]]

        strat = {"margin": margin}
        for k in a.ks:
            strat[f"f{k}_pred"] = f_pred[k]
            strat[f"f{k}_gt"] = FR[k][te]

        # GATE C — is this Table 3 restated?
        for k in a.ks:
            gatec.append(dict(fold=held, k=k,
                              rho_pred=float(stats.spearmanr(f_pred[k], margin).statistic),
                              rho_gt=float(stats.spearmanr(FR[k][te], margin).statistic)))

        preds = {}
        for arm, get in ARMS.items():
            pf, _ = _fit(get(tr), ytr, a.C, a.estimator, a.seed)
            preds[arm] = pf(get(te))

        # GATE R — regression on the standing value
        gater.append(dict(fold=held, arm="m",
                          macro_f1=float(f1_score(yte, preds["m"], average="macro",
                                                  labels=CLASSES))))

        def emit(sname, bname, sel):
            if sel.sum() < a.min_cells or len(np.unique(yte[sel])) < 2:
                return False
            base = dict(fold=held, stratifier=sname, bin=bname, n=int(sel.sum()),
                        sigma=float(SIG[te][sel].mean()),
                        cos_m_mu=float(COSMMU[te][sel].mean()),
                        cos_m_nbr=float(COSMNBR[te][sel].mean()))
            for c in CLASSES:
                base[f"frac_{c}"] = float((yte[sel] == c).mean())
            for arm in ARMS:
                r = dict(base, arm=arm,
                         macro_f1=float(f1_score(yte[sel], preds[arm][sel],
                                                 average="macro", labels=CLASSES)))
                pr, rc, _, sup = precision_recall_fscore_support(
                    yte[sel], preds[arm][sel], labels=CLASSES, zero_division=0)
                for j, c in enumerate(CLASSES):          # confound-clean within-class readout
                    r[f"prec_{c}"], r[f"rec_{c}"], r[f"sup_{c}"] = float(pr[j]), float(rc[j]), int(sup[j])
                rows.append(r)
            return True

        for sname, sval in strat.items():
            dropped = []
            for bname, sel in _levels(sval, a.nq).items():
                if not emit(sname, f"L:{bname}", sel):
                    dropped.append(f"{bname}(n={int(sel.sum())})")
            if sname.startswith("f"):
                for bname, sel in _coarse(sval).items():
                    if not emit(sname, f"C:{bname}", sel):
                        dropped.append(f"{bname}(n={int(sel.sum())})")
            if dropped:                                   # never a silent cap
                print(f"    DROPPED {sname}: {', '.join(dropped)} (< {a.min_cells} cells "
                      f"or < 2 classes)", flush=True)
        emit("ALL", "C:all", np.ones(te.sum(), bool))
        print(f"  {held}: done ({te.sum():,} cells)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_fold.tsv"), sep="\t", index=False)
    gc = pd.DataFrame(gatec); gc.to_csv(os.path.join(a.out, "gate_c.tsv"), sep="\t", index=False)
    gr = pd.DataFrame(gater)
    m_all_mean = float(df[(df.stratifier == "ALL") & (df.arm == "m")].macro_f1.mean())

    # ── paired contrasts: BOTH mu-m (the user's question) and (m+mu)-m (the standing one) ────────
    ctr = []
    for (sname, bname), g in df[df.stratifier != "ALL"].groupby(["stratifier", "bin"]):
        p = g.pivot_table(index="fold", columns="arm", values="macro_f1")
        if not {"m", "mu"}.issubset(p.columns):
            continue
        # `nbr-m` and `mu-nbr` are the two [user 2026-08-26] asked for: is the learned predictor
        # doing anything a training-free average does not?
        # ⚠ `mu-nbr` is the ONLY contrast here whose baseline is not `m`: it is the head-to-head
        # [user 2026-08-26] actually asked for. Its `m` and `mu` report columns still carry the
        # arm means, so read `delta` against the contrast NAME, never against the `m` column.
        # ★ T4: `z-nbr` and `(m+z)-(m+nbr)` are THE discriminating contrasts. `z` beating `m` is not
        #   the question -- `nbr`, a training-free arithmetic mean, already matched the v1 predictor
        #   on every well-powered axis and out-scores every `z` pooled. If z tracks nbr stratum for
        #   stratum, the bottleneck learned an average.
        for cname, hi, lo in [("mu-m", "mu", "m"), ("(m+mu)-m", "m+mu", "m"),
                              ("nbr-m", "nbr", "m"), ("(m+nbr)-m", "m+nbr", "m"),
                              ("mu-nbr", "mu", "nbr"),
                              ("z-m", "z", "m"), ("(m+z)-m", "m+z", "m"),
                              ("z-nbr", "z", "nbr"), ("(m+z)-(m+nbr)", "m+z", "m+nbr"),
                              ("z-mu", "z", "mu")]:
            if hi not in p.columns or lo not in p.columns:
                continue
            d = (p[hi] - p[lo]).dropna()
            if len(d) < 3:
                continue
            try:
                pv = float(stats.wilcoxon(p[hi].dropna(), p[lo].dropna()).pvalue)
            except Exception:
                pv = float("nan")
            sub = g[g.arm == "m"]
            ctr.append(dict(stratifier=sname, bin=bname, contrast=cname, n_folds=len(d),
                            n_cells_med=int(sub.n.median()),
                            m=round(float(p["m"].mean()), 4), mu=round(float(p["mu"].mean()), 4),
                            delta=round(float(d.mean()), 4), folds_pos=int((d > 0).sum()),
                            p=round(pv, 4), mde80=round(float(mde80(float(d.std(ddof=1)), len(d))), 4),
                            sigma=round(float(sub.sigma.mean()), 4),
                            cos_m_mu=round(float(sub.cos_m_mu.mean()), 4),
                            cos_m_nbr=round(float(sub.cos_m_nbr.mean()), 4)))
    cd = pd.DataFrame(ctr).sort_values(["stratifier", "contrast", "bin"])
    cd.to_csv(os.path.join(a.out, "contrasts.tsv"), sep="\t", index=False)

    # ── report ──────────────────────────────────────────────────────────────────────────────────
    ok = abs(m_all_mean - STANDING_M[a.estimator]) < 5e-5
    L = ["# P16 — neighbourhood homogeneity: does mu behave differently in same-type vs "
         "different-type neighbourhoods?\n",
         f"<!-- {datetime.now().isoformat(timespec='seconds')} -->\n",
         f"`[FINAL n={len(slides)}/12 slides]` · {len(Y):,} cells · mu from "
         f"`{os.path.basename(a.mu_dir)}` · estimator **{a.estimator}** · C = {a.C} FIXED · "
         f"k primary = {k_pri}\n",
         "\n## GATE R — regression on the standing control\n",
         f"arm `m` at stratifier=ALL = **{m_all_mean:.4f}**, standing = "
         f"{STANDING_M[a.estimator]:.4f} => **{'PASS' if ok else 'FAIL'}**"
         f"{'' if ok else '  ⛔ strata are unreadable; do not interpret anything below'}\n",
         "\n## GATE C — does this re-derive Table 3?\n",
         "Spearman(own-type fraction, m-probe decision margin), per fold. Table 3 stratifies on that "
         "margin. A large |rho| means this experiment is Table 3 restated, not a second result.\n",
         "| k | rho(f_pred, margin) mean | min | max | rho(f_gt, margin) mean |",
         "|---:|---:|---:|---:|---:|"]
    for k in a.ks:
        s = gc[gc.k == k]
        L.append(f"| {k} | {s.rho_pred.mean():+.3f} | {s.rho_pred.min():+.3f} | "
                 f"{s.rho_pred.max():+.3f} | {s.rho_gt.mean():+.3f} |")
    L += ["\n## PRIMARY — the user's three cases, label-free stratifier "
          f"(`f{k_pri}_pred`)\n",
          "`all_diff` = every one of the k nearest neighbours is predicted a different type from the "
          "target; `all_same` = every one the same. delta = macro-F1(arm) − macro-F1(m), paired over "
          "folds.\n",
          "| bin | contrast | n cells (med/fold) | m | mu | Δ | folds+ | p | MDE80 | sigma | "
          "cos(m,mu) | cos(m,nbr m) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pri = cd[(cd.stratifier == f"f{k_pri}_pred") & (cd.bin.str.startswith("C:"))]
    for _, r in pri.iterrows():
        L.append(f"| {r.bin[2:]} | {r.contrast} | {r.n_cells_med} | {r.m:.4f} | {r.mu:.4f} | "
                 f"{r.delta:+.4f} | {r.folds_pos}/{r.n_folds} | {r.p:.3f} | {r.mde80:.4f} | "
                 f"{r.sigma:.4f} | {r.cos_m_mu:.4f} | {r.cos_m_nbr:.4f} |")
    L += ["\n⚠ **Read the two flat-cases differently.** Flat on the ABSOLUTE `mu` column across "
          "`all_diff` / `all_same` => mu is not reading the neighbourhood => a real defect, which is "
          "the check [user] asked for. Flat on **Δ** => not a defect: m and mu degrade together and "
          "there is no complementarity.",
          "\n⚠ `f*_gt` rows in `contrasts.tsv` are a **DIAGNOSTIC**, not a competitor: that "
          "stratifier reads held-out labels, so the `all_same` stratum contains its own answer by "
          "construction. Never table it beside the label-free rows.",
          "\n## Per-slide nearest-neighbour radius (labelled cells only)\n",
          "| slide | " + " | ".join(f"median nn{k} radius (um)" for k in a.ks) + " |",
          "|---|" + "---:|" * len(a.ks)]
    for s in slides:
        L.append(f"| {s} | " + " | ".join(f"{radii[s][k]:.1f}" for k in a.ks) + " |")
    L += ["\n⚠ The labelled fraction of segmented cells ranges 56.6%-97.3% across these slides, so "
          "the k nearest **labelled** neighbours span different physical radii per slide. The table "
          "above makes that visible; it is not corrected for.",
          "\n## How to read this\n",
          "- Every stratum reports n and MDE80. MDE80 above the effect size => that stratum is "
          "UNDERPOWERED and a null there means 'no effect large enough to detect'.",
          "- `frac_*` and `prec_/rec_/sup_` columns in `per_fold.tsv` carry the class composition and "
          "the per-class readout: own-type fraction is class-confounded (Immune mean 0.560 vs Tumor "
          "0.759), so a stratum difference must be checked against them before it is called a "
          "neighbourhood effect.",
          "- macro-F1 is undefined within a single class; the within-class readout is per-class "
          "recall/precision, not macro-F1.",
          "\n## ⚠⚠ Label-ceiling confound (pre-registered before this run)\n",
          "`PROJECT_NOTES.md:464` (2026-08-19) pre-registered this confound for exactly this "
          "stratifier. The GT labels are RNA-derived and spatially smoothed, so **the label ceiling "
          "is itself probably stratum-dependent** — labels are likely MORE accurate in homogeneous "
          "(high-`f`) neighbourhoods.",
          "- A win in **mixed** regions is separable: 'context helps in mixed regions' and 'labels "
          "are worse in mixed regions' predict OPPOSITE signs.",
          "- A win in **homogeneous** regions is **partly confounded** with the labels simply being "
          "easier there, and this run CANNOT disentangle it. Report it with the confound attached.",
          "- ⚠ Nothing in this design estimates the per-stratum label ceiling."]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(L) + "\n")

    json.dump(dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        script=os.path.abspath(__file__),
        mu_dir=os.path.abspath(a.mu_dir), cache=a.cache, fov=a.fov, per_slide=a.per_slide,
        niche_dir=(os.path.abspath(a.niche_dir) if a.niche_dir else None),
        niche_arms_active=bool(a.niche_dir),
        niche_note=("z / m+z arms ON; f*_gt strata are DIAGNOSTIC for z (pure neighbourhood code, "
                    "never fitted to the target's own m), f*_pred carries any claim; the "
                    "discriminating contrasts are z-nbr and (m+z)-(m+nbr)"
                    if a.niche_dir else "z arms OFF; output bit-identical to the pre-2026-08-30 path"),
        C=a.C, nq=a.nq, ks=a.ks, seed=a.seed, estimator=a.estimator, min_cells=a.min_cells,
        nbr_k=a.nbr_k, nbr_over="all_segmented_cells",   # [user 2026-08-26] option (a)
        n_slides=len(slides), n_cells=int(len(Y)),
        gate_r=dict(measured=m_all_mean, standing=STANDING_M[a.estimator], passed=bool(ok)),
        gate_c=dict(rho_pred_mean={str(k): float(gc[gc.k == k].rho_pred.mean()) for k in a.ks},
                    note="large |rho| => this re-derives Table 3's margin stratifier"),
        nn_radius_um={s: {str(k): radii[s][k] for k in a.ks} for s in slides},
        stratifiers=dict(primary=f"f{k_pri}_pred (label-free: neighbour types predicted by the "
                                 "m-only probe fitted on the 11 TRAINING slides)",
                         diagnostic=[f"f{k}_gt" for k in a.ks] + ["reads held-out labels; "
                                                                  "mechanism check only"],
                         robustness=[f"f{k}_pred" for k in a.ks[1:]]),
        preregistration=dict(
            primary="macro-F1(mu) - macro-F1(m), paired over 12 folds, f_pred==0 vs f_pred==1",
            shape="the two extremes should differ [user 2026-08-25]",
            flat_absolute="=> real defect: mu is not reading the neighbourhood",
            flat_delta="=> not a defect: m and mu degrade together, no complementarity",
            confound="own-type fraction is class-confounded (eta^2 0.064); within-class readout "
                     "is per-class recall/precision"),
    ), open(os.path.join(a.out, "PROVENANCE.json"), "w"), indent=2)

    print(f"\nGATE R: m={m_all_mean:.4f} vs standing {STANDING_M[a.estimator]:.4f} "
          f"=> {'PASS' if ok else 'FAIL'}")
    for k in a.ks:
        print(f"GATE C: k={k} rho(f_pred, margin) = {gc[gc.k == k].rho_pred.mean():+.3f}")
    print(f"\nwrote {a.out}/report.md")
    print(pri.to_string(index=False))
    if not ok and not a.skip_gate:
        sys.exit(3)


if __name__ == "__main__":
    main()
