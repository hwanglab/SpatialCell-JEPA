#!/usr/bin/env python3
"""probe_nbr_composition.py — does the cell's OWN embedding encode its neighbours' cell types? (P16 T2, camera-ready)

[user 2026-11-05] "yes" to the neighbour-composition probe (plan: results/reports/20261105_camera_ready_finalize_plan/plan.md).
Tests the title literally, independent of the cell's own type: decode the reference-type COMPOSITION of a cell's 8 nearest
labelled neighbours from (a) m, its embedding as used in the paper, and (b) m_iso(r), the same cell embedded with its
neighbourhood tokens dropped (cache_isolated_cell.py). m − m_iso = neighbourhood information the embedding carries beyond
what the isolated cell (incl. its own type, via homophily) implies.

Cells   : the 144,000 typing cells (results/masked_ctx/eval12_typing_cells.parquet; 12,000 / slide, 12 HD slides).
Target  : y = fraction of Immune / Stromal / Tumor among the 8 nearest qc_pass 3-class labelled cells, self excluded
          (k = 8 and the population = neighborhood_homogeneity_strata.py, the paper's Appendix definition).
Model   : RidgeCV (alphas 10^-1..10^4, 6 values, efficient LOO on the TRAINING folds only), features standardised on the
          training fold. Multi-output.
Split   : leave-one-patient-out, 12 folds (fit on 11 slides, score the 12th).
Metric  : R² per class on the held-out slide (sklearn r2_score; < 0 = worse than predicting the held-out slide's mean),
          macro = mean over the 3 classes. Paired Δ per fold.
Arms    : m · m_iso(r) for each r · DIAGNOSTIC own_type = one-hot of the cell's OWN reference label (reads held-out labels;
          NOT a deployable method: it is the homophily floor, what the cell's type alone predicts about its neighbours).
PRE-REGISTERED (fixed before any number): PRIMARY m − m_iso(4) macro-R² ≥ +0.02 AND ≥ 8/12 folds
          => "the embedding encodes neighbourhood composition beyond the isolated cell".
"""
import os, json, time, argparse, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

CLASSES = ["Immune", "Stromal", "Tumor"]
K = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache_fm_hd/hoptimus0")
    ap.add_argument("--iso-root", default="cache/isocell")
    ap.add_argument("--rings", default="4,8,14")
    ap.add_argument("--labels", default="results/eval/gt_labels_hd12")
    ap.add_argument("--cells", default="results/masked_ctx/eval12_typing_cells.parquet")
    ap.add_argument("--out", default="results/reports/20261105_t2_isocell_typing/nbr_composition")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    rings = [x.strip() for x in a.rings.split(",")]
    tc = pd.read_parquet(a.cells)
    slides = sorted(tc.slide.unique())
    srcs = {"m": a.cache, **{f"m_iso_r{r}": f"{a.iso_root}/r{r}" for r in rings}}
    X = {k: [] for k in srcs}; Y, OWN, S = [], [], []
    for s in slides:
        ri = np.sort(tc[tc.slide == s].row_idx.values.astype(np.int64))
        lab = pd.read_parquet(f"{a.labels}/{s}.parquet")
        lab = lab[lab.qc_pass & lab.L1.isin(CLASSES)]
        idx_full = lab.row_idx.to_numpy()
        y_full = pd.Categorical(lab.L1, categories=CLASSES).codes.astype(np.int8)
        pos = {r: i for i, r in enumerate(idx_full)}
        assert all(r in pos for r in ri), f"{s}: typing cell not in the labelled population"
        sel = np.array([pos[r] for r in ri])
        z0 = np.load(f"{a.cache}/{s}/fov56/cells.npz")
        xy = z0["centroid_xy"][idx_full].astype(np.float64)
        nn = cKDTree(xy).query(xy[sel], k=K + 1)[1]
        comp = np.zeros((len(sel), 3), np.float32)
        for row, (i, nb) in enumerate(zip(sel, nn)):
            nb = nb[nb != i][:K]
            comp[row] = np.bincount(y_full[nb], minlength=3) / K
        Y.append(comp); OWN.append(np.eye(3, dtype=np.float32)[y_full[sel]]); S += [s] * len(sel)
        for k, d in srcs.items():
            X[k].append(np.load(f"{d}/{s}/fov56/cells.npz")["m_cell"][ri].astype(np.float32))
        print(f"  {s}: n={len(sel)} mean comp={comp.mean(0).round(3).tolist()}", flush=True)
    Y = np.concatenate(Y); OWN = np.concatenate(OWN); S = np.array(S)
    X = {k: np.concatenate(v) for k, v in X.items()}
    X["own_type_DIAGNOSTIC"] = OWN
    rows = []
    for held in slides:
        tr, te = S != held, S == held
        for arm, F in X.items():
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
            mdl = RidgeCV(alphas=np.logspace(-1, 4, 6)).fit((F[tr] - mu) / sd, Y[tr])
            P = mdl.predict((F[te] - mu) / sd)
            r2 = [r2_score(Y[te][:, c], P[:, c]) for c in range(3)]
            rows.append(dict(fold=held, arm=arm, alpha=float(mdl.alpha_), r2_macro=float(np.mean(r2)),
                             **{f"r2_{c}": float(v) for c, v in zip(CLASSES, r2)}))
        print(f"  fold {held} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}/per_fold.tsv", sep="\t", index=False)
    piv = df.pivot(index="fold", columns="arm", values="r2_macro")

    def pair(A, B):
        d = piv[A] - piv[B]
        return dict(A=A, B=B, A_mean=piv[A].mean(), B_mean=piv[B].mean(), delta=d.mean(), folds_plus=int((d > 0).sum()),
                    p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    comps = [pair("m", f"m_iso_r{r}") for r in rings] + [pair("m", "own_type_DIAGNOSTIC")] + \
            [pair(f"m_iso_r{r}", "own_type_DIAGNOSTIC") for r in rings]
    L = ["# Neighbour-composition probe — does the cell's own embedding encode its neighbours' types?\n",
         "Target: fraction Immune/Stromal/Tumor among the 8 nearest labelled cells (self excluded). RidgeCV (alpha by LOO on "
         "training folds), standardised; LOPO 12 folds over HD-12; 12,000 typing cells/slide. Metric: macro R² over the 3 "
         "classes on the held-out slide (<0 = worse than the held-out mean). own_type_DIAGNOSTIC reads the held-out cell's own "
         "reference label: a homophily floor, NOT a method.\n",
         "| arm | mean macro R² | R² Immune | R² Stromal | R² Tumor |", "|---|---|---|---|---|"]
    for arm in X:
        g = df[df.arm == arm]
        L.append(f"| {arm} | {g.r2_macro.mean():.4f} | {g.r2_Immune.mean():.4f} | {g.r2_Stromal.mean():.4f} | {g.r2_Tumor.mean():.4f} |")
    L += ["", "| A | B | A | B | Δ | folds A>B | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for c in comps:
        L.append(f"| {c['A']} | {c['B']} | {c['A_mean']:.4f} | {c['B_mean']:.4f} | {c['delta']:+.4f} | {c['folds_plus']}/12 | {c['p']:.3g} |")
    p0 = comps[0]
    L.append(f"\nPRIMARY (m − m_iso_r{rings[0]} ≥ +0.02, ≥8/12): "
             f"{'BAR CLEARED' if p0['delta'] >= 0.02 and p0['folds_plus'] >= 8 else 'BAR NOT CLEARED'} ({p0['delta']:+.4f}, {p0['folds_plus']}/12)")
    open(f"{a.out}/REPORT.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": os.path.abspath(__file__), "args": vars(a),
               "inputs": {k: v for k, v in srcs.items()}, "n_cells": int(len(Y)), "wall_s": time.time() - t0},
              open(f"{a.out}/PROVENANCE.json", "w"), indent=1)


if __name__ == "__main__":
    main()
