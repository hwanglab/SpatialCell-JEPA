#!/usr/bin/env python3
"""
probe_backbone_fov_grid.py — {backbone} x {FOV} x {LOPO, random} cell-typing probe on the 12 HD slides.

THE QUESTION (stage 2 of the 2026-08-07 plan). Three downstream nulls — typing, DE marker strength,
CAF-subtype recovery — were all measured on Virchow2, which the geometry diagnostics then showed to
be a dramatic outlier: effective rank 12.3 vs H-optimus-0's 204.2, PC1 share 0.523 vs 0.090,
neighbour-R2 0.430 vs 0.153 on IDENTICAL cells. So "the method adds nothing" and "this backbone has
no room" are confounded. This probe separates them.

★ THE DECISIVE READ is not which backbone wins. It is the INTERACTION:
    does a wider FOV RAISE the random-split score while LOWERING the LOPO score?
  If yes, context helps within a slide and hurts transfer — which would (a) explain the standing
  LOPO 0.580 vs random-split 0.727 gap as context-riding rather than a probe failure, and (b) mean
  the 2026-07-21 promotion of the target FOV from 56 to 112 um traded transfer for within-slide fit.

WHAT MAKES THIS A CLEAN CONTRAST (verified 2026-08-07, all 12 slides x fov{56,112}):
`centroid_xy` is md5-identical between `cache_ms_frozen` (Virchow2, d=1280) and
`cache_fm_hd/hoptimus0` (d=1536) — same cells, same row order, 2,958,229 total. Every arm therefore
sees the SAME cells, the SAME subsample rows, the SAME folds and the SAME seed; only the feature
matrix changes. The per-slide subsample is drawn ONCE, before any arm is built.

LOPO IS PATIENT-OUT. The 12 HD slides carry 12 distinct `patient_id`s in the canonical
`per_slide.tsv` (looked up 2026-08-07, not derived from the slide names).

PROBE = the project's standard one, unchanged so the numbers stay comparable to every published
arm: multinomial LogisticRegression(C=1.0, max_iter=300, class_weight='balanced') on a
StandardScaler fitted INSIDE each fold. Classes default to {Tumor, Immune, Stromal} — the 3-class
vocabulary behind the published 0.580 / 0.727.

Labels: results/eval/gt_labels_hd12/<slide>.parquet (attach_gt_hd12.py, EXACT object_id join).
⚠ Those carry `row_idx` into the polygons_v2 caches. They are NOT interchangeable with
results/eval/gt_labels/*.parquet, whose `cache_idx` indexes the HoverFast whole-slide population.

USAGE
  stage 2 :  --fovs 56,112
  stage 3 :  --fovs 56,112,168 --concat 56+112,56+112+168     (the training-free multi-FOV control)
  bridge  :  --slides <three slides>  (comparability with the 3-slide numbers)

⚠ CLUSTER ONLY. Holds several [N, 1280..4352] float32 matrices and fits ~50 multinomial
logistic regressions; an earlier interactive run of this class left an orphaned process at 678% CPU
on a shared login node for 41 minutes.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P16 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(P16, "scripts", "eval"))
from celltype_probe_hd import ece                        # identical ECE definition  # noqa: E402
# --- anonymized slide list (replaces `from gate_n12_polygons import SLIDES`) ------------
# Only element [0] of each tuple is ever read. Slides are numbered as in the paper's
# slide-composition appendix; the original accession numbers are not part of this release.
SLIDES = [("Slide 6", "Slide 6", "A"), ("Slide 2", "Slide 2", "A"), ("Slide 12", "Slide 12", "A"), ("Slide 4", "Slide 4", "A"), ("Slide 5", "Slide 5", "A"), ("Slide 3", "Slide 3", "A"), ("Slide 7", "Slide 7", "A"), ("Slide 8", "Slide 8", "A"), ("Slide 1", "Slide 1", "A"), ("Slide 11", "Slide 11", "A"), ("Slide 10", "Slide 10", "A"), ("Slide 9", "Slide 9", "A")]
# ---------------------------------------------------------------------------------------

GT = "results/eval/gt_labels_hd12"
SEED = 0
PCA_N = 0          # set from --pca in main(); 0 = OFF = bit-identical to the pre-2026-08-19 path
DEFAULT_ROOTS = "virchow2:results/cache_ms_frozen,hoptimus0:results/cache_fm_hd/hoptimus0"


def neighbour_mean(m_full, xy_full, idx, k, batch=4096):
    """Mean embedding of each TARGET cell's k nearest OTHER cells.

    The k-d tree is built on ALL cells in the cache, not just the labelled subsample, so the
    neighbourhoods are the real ones. Self is dropped exactly as in
    backbone_geometry_diagnostics.neighbour_predictability (query k+2, drop self, take first k),
    which is also the rule apply_predictor_hd uses — so this smoother sees precisely the context
    the predictor sees. That is the whole point of the control: it is the training-free version of
    what the predictor does, with no cell-JEPA in it.

    Coordinates stay in raw cache pixels: k-NN RANKING is invariant to a global scale factor, so no
    mpp conversion is needed — and the cohort mpp is NOT a constant (0.2522 vs 0.2525 on the B2
    re-cuts), so not needing it is one less thing to get wrong.
    """
    tree = cKDTree(xy_full)
    out = np.empty((len(idx), m_full.shape[1]), dtype=np.float32)
    for b0 in range(0, len(idx), batch):
        t = idx[b0:b0 + batch]
        _, nn = tree.query(xy_full[t], k=k + 2, workers=-1)
        ctx = np.empty((len(t), k), dtype=np.int64)
        for i, (ti, row) in enumerate(zip(t, nn)):
            ctx[i] = row[row != ti][:k]
        out[b0:b0 + len(t)] = m_full[ctx].mean(1)
    return out


def load(slides, roots, fovs, keep, per_slide, require_qc, nbr_k=0, pred_out=None, pred_fov=None):
    """-> y[slide], X[slide][(label,fov)], rows[slide]. Subsample drawn ONCE and shared by all arms.

    ⚠ The rng is touched ONLY by the per-slide subsample draw below. `nbr_k` and `pred_out` add no
    rng call and change no existing key, so the m-only arms stay bit-identical to the
    pre-2026-08-09 runs — which is what makes them usable as a regression gate against the
    published 0.6201 / 0.6080.

    `pred_out` = {label: predictor_out_dir} (added 2026-08-09 for the A1 ceiling eval). μ is read
    from <dir>/<slide>.npz and indexed by the SAME `row_idx` used for m_cell. That join is only
    valid because apply_predictor_hd.py writes μ in FULL CACHE-CELL ORDER over the same
    polygons_v2 population that `row_idx` indexes — asserted below on length, not assumed.
    Only the FOV equal to the predictor's --fine-fov gets a μ (that is the scale it predicts).
    """
    rng = np.random.default_rng(SEED)
    y, X, rows = {}, {}, {}
    for s in slides:
        lab = pd.read_parquet(os.path.join(GT, f"{s}.parquet"))
        lab = lab[lab.L1.isin(keep)]
        if require_qc:
            lab = lab[lab.qc_pass]
        lab = lab.reset_index(drop=True)
        take = np.arange(len(lab))
        if per_slide and len(take) > per_slide:
            take = np.sort(rng.choice(take, per_slide, replace=False))
        lab = lab.iloc[take]
        idx = lab.row_idx.to_numpy()                     # positions in the polygons_v2 cache
        y[s] = lab.L1.to_numpy()
        rows[s] = idx
        X[s] = {}
        for label, root in roots:
            for fov in fovs:
                p = os.path.join(root, s, f"fov{fov}", "cells.npz")
                z = np.load(p)
                assert len(z["centroid_xy"]) > idx.max(), f"{p}: row_idx out of range for this cache"
                if nbr_k:
                    mf = np.asarray(z["m_cell"], dtype=np.float32)   # full slide, needed for the tree
                    X[s][(label, fov)] = mf[idx].copy()
                    X[s][(label, fov, "nbr")] = neighbour_mean(
                        mf, np.asarray(z["centroid_xy"], dtype=np.float64), idx, nbr_k)
                    del mf
                else:
                    X[s][(label, fov)] = np.asarray(z["m_cell"][idx], dtype=np.float32)
                if pred_out and label in pred_out and int(fov) == int(pred_fov):
                    pp = os.path.join(pred_out[label], f"{s}.npz")
                    assert os.path.exists(pp), f"MISSING predictor output: {pp}"
                    zp = np.load(pp)
                    assert len(zp["mu"]) == len(z["centroid_xy"]), (
                        f"{pp}: mu has {len(zp['mu']):,} rows but the cache has "
                        f"{len(z['centroid_xy']):,} — NOT the same cell population/order, so "
                        f"row_idx would silently attach the wrong mu to every cell")
                    assert zp["mu"].shape[1] == z["m_cell"].shape[1], (
                        f"{pp}: mu dim {zp['mu'].shape[1]} != m_cell dim {z['m_cell'].shape[1]}")
                    X[s][(label, fov, "mu")] = np.asarray(zp["mu"][idx], dtype=np.float32)
                    del zp
                del z
    return y, X, rows


def fit_eval(Xtr, ytr, Xte, yte, classes):
    """2026-08-19 [user] added the module-level PCA_N (`--pca N`), lifted VERBATIM from
    `p2_typing_and_slideid.py::fit_score` (added there 2026-08-15 by [user] for the same reason).

    WHY IT MATTERS HERE SPECIFICALLY: the `m+nbrK` arms are 3072-d concatenations of two
    near-collinear 1536-d blocks, and StandardScaler gives all 3072 dims equal weight — the exact
    conditioning failure the 08-15 flag was introduced to attack on `m+mu`. It also removes a real
    CONFOUND in this sweep: at PCA_N=0 the `m` arm is 1536-d and `m+nbrK` is 3072-d, so the contrast
    mixes "added neighbour information" with "doubled dimension at fixed C". At PCA_N=64 both arms
    enter the probe at 64 dims and only the information differs.

    ⚠ FIT ON TRAIN ONLY — fitting on the held-out slide would leak.
    ⚠ PCA_N = 0 keeps this path bit-identical to the pre-2026-08-19 code: the only change at 0 is
    that the two `sc.transform` results are materialised into names instead of passed inline, which
    is the same operation in the same order.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if PCA_N and PCA_N > 0:
        from sklearn.decomposition import PCA
        nc = min(PCA_N, Ztr.shape[1], Ztr.shape[0] - 1)
        pc = PCA(n_components=nc, svd_solver="randomized", random_state=SEED).fit(Ztr)
        Ztr, Zte = pc.transform(Ztr), pc.transform(Zte)
    clf = LogisticRegression(max_iter=300, C=1.0, multi_class="multinomial",
                             class_weight="balanced", n_jobs=-1).fit(Ztr, ytr)
    pr = clf.predict_proba(Zte)
    cl = list(clf.classes_)
    pred = clf.classes_[pr.argmax(1)]
    return (dict(n_train=int(len(ytr)), n_test=int(len(yte)),
                 macro_f1=float(f1_score(yte, pred, average="macro", labels=classes)),
                 acc=float(accuracy_score(yte, pred)),
                 ece=ece(pr, np.array([cl.index(v) for v in yte])),
                 per_class_f1={c: float(v) for c, v in
                               zip(classes, f1_score(yte, pred, average=None, labels=classes))}),
            confusion_matrix(yte, pred, labels=classes))


def run_lopo(build, y, slides, classes):
    folds, conf = [], np.zeros((len(classes),) * 2, np.int64)
    Xby = {s: build(s) for s in slides}
    for held in slides:
        tr = [s for s in slides if s != held]
        f, c = fit_eval(np.concatenate([Xby[s] for s in tr], 0),
                        np.concatenate([y[s] for s in tr], 0),
                        Xby[held], y[held], classes)
        f["held"] = held
        folds.append(f); conf += c
        print(f"      fold {held:18s} macro-F1={f['macro_f1']:.4f}", flush=True)
    return folds, conf, int(next(iter(Xby.values())).shape[1])


def run_random(build, y, slides, classes, tr_idx, te_idx):
    X = np.concatenate([build(s) for s in slides], 0)     # pooled in `slides` order
    yy = np.concatenate([y[s] for s in slides], 0)
    f, c = fit_eval(X[tr_idx], yy[tr_idx], X[te_idx], yy[te_idx], classes)
    f["held"] = "random30"
    return [f], c, int(X.shape[1])


def main():
    os.chdir(P16)
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=DEFAULT_ROOTS, help="comma list of label:path")
    ap.add_argument("--fovs", default="56,112")
    ap.add_argument("--concat", default="", help="e.g. 56+112,56+112+168 — training-free multi-FOV arms")
    ap.add_argument("--neighbour-concat", type=int, default=0, metavar="K",
                    help="K>0 adds two TRAINING-FREE arms per (backbone,FOV): 'm+nbrK' = the cell "
                         "concatenated with the mean of its K nearest neighbours, and 'nbrKonly' = "
                         "that neighbour mean alone. K=36 mirrors the predictor's own context size, "
                         "so this is the no-cell-JEPA control for 'does neighbour context add "
                         "anything at all'. Off (0) by default; changes no existing arm.")
    ap.add_argument("--predictor-out", default="",
                    help="comma list of label:predictor_out_dir (e.g. "
                         "'virchow2:results/eval/predictor_out_ceiling_virchow2'). Adds "
                         "'m+mu' = the cell's own embedding concatenated with the predictor's "
                         "mu, and 'muonly' = mu alone. ⚠ mu is joined by the SAME row_idx as "
                         "m_cell and is only valid for the FOV the predictor was trained to "
                         "predict (--predictor-fov).")
    ap.add_argument("--predictor-fov", type=int, default=112,
                    help="the predictor's --fine-fov, i.e. the scale mu lives at. Default 112 "
                         "(PROJECT_NOTES 2026-07-21 canonical target_fov).")
    ap.add_argument("--splits", default="lopo,random")
    ap.add_argument("--slides", default="", help="comma list; default = canonical 12")
    ap.add_argument("--keep-classes", default="Tumor,Immune,Stromal")
    ap.add_argument("--per-slide", type=int, default=25000, help="0 = all labelled cells")
    ap.add_argument("--skip-single", action="store_true",
                    help="build ONLY the --concat arms. Lets an unattended job run the make-or-break "
                         "multi-FOV control FIRST, so a wall timeout cannot cost the answer; the "
                         "single-scale reference arms then run as a separate, losable invocation.")
    ap.add_argument("--require-qc", action="store_true",
                    help="restrict to qc_pass cells (OFF by default: the published 3-slide numbers "
                         "did not filter on it, and comparability is the point of the bridge run)")
    ap.add_argument("--pca", type=int, default=0, metavar="N",
                    help="N>0 => randomized PCA to N components, FIT ON THE TRAINING SPLIT ONLY and "
                         "applied to both sides, after StandardScaler. Same implementation as "
                         "p2_typing_and_slideid.py::fit_score (added there 2026-08-15 by [user]). "
                         "Attacks the conditioning of wide concat arms (m+nbrK is 3072-d, two "
                         "near-collinear 1536-d blocks) AND removes the 1536-vs-3072 dimension "
                         "confound between the m and m+nbrK arms. 0 (default) = OFF, bit-identical "
                         "to the pre-2026-08-19 path, so the standing regression gates still hold.")
    ap.add_argument("--out", default="results/reports")
    ap.add_argument("--tag", default="t2_stage2_backbone_fov_probe")
    a = ap.parse_args()

    global PCA_N
    PCA_N = int(a.pca)
    if PCA_N:
        print(f"PCA ON: {PCA_N} components, fit on TRAIN split only, random_state={SEED}")

    roots = [tuple(x.split(":", 1)) for x in a.roots.split(",") if x.strip()]
    fovs = [int(x) for x in a.fovs.split(",") if x.strip()]
    concats = [[int(v) for v in c.split("+")] for c in a.concat.split(",") if c.strip()]
    slides = ([x.strip() for x in a.slides.split(",") if x.strip()]
              or [s for s, _, _ in SLIDES])
    keep = [x.strip() for x in a.keep_classes.split(",") if x.strip()]
    splits = [x.strip() for x in a.splits.split(",") if x.strip()]

    for label, root in roots:                             # fail before any compute
        for s in slides:
            for fov in fovs:
                p = os.path.join(root, s, f"fov{fov}", "cells.npz")
                assert os.path.exists(p), f"MISSING {label} cache: {p}"

    t0 = time.time()
    pred_out = dict(kv.split(":", 1) for kv in a.predictor_out.split(",") if kv.strip())
    if pred_out:
        assert a.predictor_fov in fovs, (
            f"--predictor-fov {a.predictor_fov} not in --fovs {fovs}: mu would be loaded for no arm")
        for lb, d in pred_out.items():
            assert lb in dict(roots), f"--predictor-out label '{lb}' is not a --roots label"
            for s in slides:
                assert os.path.exists(os.path.join(d, f"{s}.npz")), f"MISSING {d}/{s}.npz"
        print(f"predictor-out arms: {pred_out} at fov{a.predictor_fov}")
    y, X, rows = load(slides, roots, fovs, set(keep), a.per_slide, a.require_qc, a.neighbour_concat,
                      pred_out=pred_out, pred_fov=a.predictor_fov)
    classes = sorted(set(np.concatenate([y[s] for s in slides])))
    n_tot = sum(len(y[s]) for s in slides)
    print(f"loaded {n_tot:,} labelled cells over {len(slides)} slides, classes={classes} "
          f"({time.time()-t0:.0f}s)")
    for s in slides:
        print(f"  {s:18s} n={len(y[s]):>7,}  {dict(pd.Series(y[s]).value_counts())}")

    # one pooled stratified split, shared by every arm
    tr_idx = te_idx = None
    if "random" in splits:
        from sklearn.model_selection import train_test_split
        ypool = np.concatenate([y[s] for s in slides], 0)
        tr_idx, te_idx = train_test_split(np.arange(len(ypool)), test_size=0.30,
                                          stratify=ypool, random_state=SEED)
        print(f"  random split: {len(tr_idx):,} train / {len(te_idx):,} test (seed {SEED})")

    builders = {}
    for label, _ in roots:
        if not a.skip_single:
            for fov in fovs:
                builders[f"{label}_fov{fov}"] = (lambda s, l=label, f=fov: X[s][(l, f)])
        for cc in concats:
            if all((label, f) in X[slides[0]] for f in cc):
                nm = f"{label}_concat" + "+".join(str(f) for f in cc)
                builders[nm] = (lambda s, l=label, c=cc:
                                np.concatenate([X[s][(l, f)] for f in c], axis=1))

    # Neighbour arms LAST, and 'm+nbr' (the question) before 'nbr-only' (the diagnostic), so that a
    # wall timeout costs the least informative arm. The m-only arms above are already ahead of both:
    # they are the regression gate, and a mismatch there invalidates everything after it.
    if a.neighbour_concat:
        K = a.neighbour_concat
        for kind in ("m+nbr", "nbronly"):
            for label, _ in roots:
                for fov in fovs:
                    if (label, fov, "nbr") not in X[slides[0]]:
                        continue
                    if kind == "m+nbr":
                        builders[f"{label}_fov{fov}_m+nbr{K}"] = (
                            lambda s, l=label, f=fov:
                            np.concatenate([X[s][(l, f)], X[s][(l, f, "nbr")]], axis=1))
                    else:
                        builders[f"{label}_fov{fov}_nbr{K}only"] = (
                            lambda s, l=label, f=fov: X[s][(l, f, "nbr")])

    # Predictor (cell-JEPA mu) arms LAST, same ordering logic as the neighbour block: 'm+mu' is the
    # question, 'muonly' the diagnostic, and both sit behind the m-only regression gate.
    if pred_out:
        f = int(a.predictor_fov)
        for kind in ("m+mu", "muonly"):
            for label, _ in roots:
                if (label, f, "mu") not in X[slides[0]]:
                    continue
                if kind == "m+mu":
                    builders[f"{label}_fov{f}_m+mu"] = (
                        lambda s, l=label, ff=f:
                        np.concatenate([X[s][(l, ff)], X[s][(l, ff, "mu")]], axis=1))
                else:
                    builders[f"{label}_fov{f}_muonly"] = (
                        lambda s, l=label, ff=f: X[s][(l, ff, "mu")])

    arms = {}
    for name, build in builders.items():
        arms[name] = {}
        for sp in splits:
            print(f"  [{name}] {sp} ...", flush=True)
            t1 = time.time()
            if sp == "lopo":
                folds, conf, dim = run_lopo(build, y, slides, classes)
            else:
                folds, conf, dim = run_random(build, y, slides, classes, tr_idx, te_idx)
            mf1 = np.array([f["macro_f1"] for f in folds])
            arms[name][sp] = dict(dim=dim, folds=folds, confusion=conf.tolist(),
                                  macro_f1_mean=float(mf1.mean()), macro_f1_std=float(mf1.std()),
                                  acc_mean=float(np.mean([f["acc"] for f in folds])),
                                  ece_mean=float(np.mean([f["ece"] for f in folds])),
                                  wall_s=round(time.time() - t1, 1))
            print(f"    -> macro-F1 {mf1.mean():.4f} +/- {mf1.std():.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(a.out, f"{ts}_{a.tag}")
    os.makedirs(outdir, exist_ok=True)
    summary = dict(generated=ts, script=os.path.abspath(__file__), roots=dict(roots), fovs=fovs,
                   concat=[("+".join(map(str, c))) for c in concats], splits=splits, slides=slides,
                   classes=classes, keep_classes=keep, per_slide=a.per_slide,
                   require_qc=bool(a.require_qc), seed=SEED, n_cells=int(n_tot),
                   neighbour_concat=int(a.neighbour_concat), pca=PCA_N,
                   predictor_out=pred_out, predictor_fov=(a.predictor_fov if pred_out else None),
                   labels=GT, probe=("LogisticRegression(multinomial, C=1.0, max_iter=300, balanced)"
                                     + (f" + PCA{PCA_N} fit on train only" if PCA_N else "")),
                   note="Same cells / rows / folds / seed across every arm; cross-backbone row order "
                        "verified md5-identical on centroid_xy for all 12 slides x fov{56,112}.",
                   arms=arms)
    json.dump(summary, open(os.path.join(outdir, "results.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(outdir, "PROVENANCE.json"), "w"), indent=2)

    L = [f"# P16 T2 — backbone x FOV x split probe ({len(slides)} HD slides)\n",
         f"_generated {ts} · classes={classes} · {n_tot:,} cells · {a.per_slide or 'all'}/slide · "
         f"seed {SEED} · LOPO = patient-out (12 distinct patient_id)_\n",
         "| arm | dim | " + " | ".join(f"{s} macro-F1" for s in splits) + " |",
         "|---|---|" + "---|" * len(splits)]
    for name, r in arms.items():
        cells = [f"{r[s]['macro_f1_mean']:.4f} ± {r[s]['macro_f1_std']:.4f}" for s in splits]
        L.append(f"| {name} | {r[splits[0]]['dim']} | " + " | ".join(cells) + " |")
    if {"lopo", "random"} <= set(splits):
        L.append("\n## The decisive read — FOV effect per backbone (transfer vs within-slide)\n")
        L.append("| backbone | FOV | LOPO | random-30 | gap (random − LOPO) |")
        L.append("|---|---|---|---|---|")
        for label, _ in roots:
            for fov in fovs:
                k = f"{label}_fov{fov}"
                if k in arms:
                    lo, rd = arms[k]["lopo"]["macro_f1_mean"], arms[k]["random"]["macro_f1_mean"]
                    L.append(f"| {label} | {fov} | {lo:.4f} | {rd:.4f} | {rd-lo:+.4f} |")
        L.append("\nContext-riding predicts: **wider FOV → random up, LOPO down → gap widens.**")

    # The pre-registered read for the neighbour control is PAIRED and LOPO-only. It is emitted here
    # rather than computed by hand afterwards so the comparison cannot be chosen after seeing the
    # numbers. Random-30 is deliberately NOT the read: concatenating neighbour context is exactly
    # what inflates a within-slide split via context-riding.
    if a.neighbour_concat and "lopo" in splits:
        K = a.neighbour_concat
        from scipy import stats as _st
        L.append(f"\n## Pre-registered read — does `m ⊕ nbr{K}` raise **LOPO**? (paired, per fold)\n")
        L.append("Every arm shares the same cells, folds and seed, so this is a PAIRED test: the "
                 "relevant scale is the SD of the per-fold delta, not the between-fold SD.\n")
        L.append(f"| backbone | FOV | m | m+nbr{K} | paired Δ | SD(Δ) | folds won | p (paired t) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for label, _ in roots:
            for fov in fovs:
                kb, kn = f"{label}_fov{fov}", f"{label}_fov{fov}_m+nbr{K}"
                if not ({kb, kn} <= set(arms)):
                    continue
                base = {f["held"]: f["macro_f1"] for f in arms[kb]["lopo"]["folds"]}
                new = {f["held"]: f["macro_f1"] for f in arms[kn]["lopo"]["folds"]}
                common = [h for h in base if h in new]
                d = np.array([new[h] - base[h] for h in common])
                p = float(_st.ttest_rel([new[h] for h in common],
                                        [base[h] for h in common]).pvalue) if len(d) > 1 else float("nan")
                L.append(f"| {label} | {fov} | {np.mean([base[h] for h in common]):.4f} | "
                         f"{np.mean([new[h] for h in common]):.4f} | {d.mean():+.4f} | "
                         f"{d.std(ddof=1):.4f} | {int((d > 0).sum())}/{len(d)} | {p:.3f} |")
        L.append(f"\n**Proceed unless the paired delta is negative and significant** "
                 f"(p<0.05 with ≥8/{len(slides)} folds negative). A positive delta says neighbour "
                 f"context carries typing information that the cell's own embedding does not — "
                 f"training-free, with no cell-JEPA in it.")
    # A1 CEILING read. Same paired, LOPO-only construction as the neighbour block above, emitted by
    # the script so the comparison cannot be chosen after seeing the numbers.
    if pred_out and "lopo" in splits:
        f = int(a.predictor_fov)
        from scipy import stats as _st2
        L.append("\n## A1 read — does `m ⊕ μ` raise **LOPO** over raw `m`? (paired, per fold)\n")
        L.append("⚠⚠ **CEILING, NOT TRANSFER.** These predictors were trained on all 12 HD slides "
                 "and applied to those same 12, so every evaluated cell was in predictor training. "
                 "Leakage can only inflate `m ⊕ μ` ⇒ a FAILURE to beat raw `m` here is decisive "
                 "against the construction; a pass is only permission to spend on the clean "
                 "84-corpus encode.\n")
        L.append(f"| backbone | FOV | m | m+μ | paired Δ | SD(Δ) | folds won | p (paired t) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for label, _ in roots:
            kb, kn = f"{label}_fov{f}", f"{label}_fov{f}_m+mu"
            if not ({kb, kn} <= set(arms)):
                continue
            base = {fo["held"]: fo["macro_f1"] for fo in arms[kb]["lopo"]["folds"]}
            new = {fo["held"]: fo["macro_f1"] for fo in arms[kn]["lopo"]["folds"]}
            common = [h for h in base if h in new]
            d = np.array([new[h] - base[h] for h in common])
            p = float(_st2.ttest_rel([new[h] for h in common],
                                     [base[h] for h in common]).pvalue) if len(d) > 1 else float("nan")
            L.append(f"| {label} | {f} | {np.mean([base[h] for h in common]):.4f} | "
                     f"{np.mean([new[h] for h in common]):.4f} | {d.mean():+.4f} | "
                     f"{d.std(ddof=1):.4f} | {int((d > 0).sum())}/{len(d)} | {p:.3f} |")
        L.append("\n⚠ ASYMMETRIC TRAINING: the Virchow2 ceiling predictor early-stopped (epoch 183 "
                 "of 200) while the H-optimus-0 one ran all 200 and was still improving — so the "
                 "H-optimus-0 arm is relatively UNDER-trained and the backbone delta is biased "
                 "against it. Do not read the cross-backbone difference without this caveat.")
    open(os.path.join(outdir, "report.md"), "w").write("\n".join(L) + "\n")
    print(f"\nwrote {outdir}/{{results.json, report.md, PROVENANCE.json}}  "
          f"total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
