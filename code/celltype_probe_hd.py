#!/usr/bin/env python3
"""
celltype_probe_hd.py — P16 Phase 5 downstream: patient-out cell-typing on 3 HD slides.

The grounding sanity ablation (design card §2.5/§10): does the predictability-augmented cell-JEPA
representation [m_cell@56 ⊕ μ ⊕ σ²] match/beat the raw frozen-Virchow2 baseline at image-only,
leave-one-slide-out cell typing? Same cells / labels / folds for every feature set -> clean head-to-head.

Feature sets (built from what is present on disk):
  raw56     : frozen Virchow2 m_cell@56              (1280)  [baseline; runnable now]
  cellJEPA  : [ m56 ⊕ μ ⊕ σ² ]                       (3840)  [needs --predictor-out; after training]
  (--ablations adds:  mu_only(1280), s2_only(1280), resid=m56-μ(1280))

Labels: results/eval/gt_labels/<slide>.parquet from attach_gt_hd.py (coarse L1, {Tumor,Immune,Stromal,NTE}).
Features: results/cache_pilot/<slide>/fov56/cells.npz['m_cell'][cache_idx]; predictor μ,σ² from
          results/eval/predictor_out/<slide>.npz (mu[N,1280], s2[N,1280], cache-cell order) if given.

Probe: LogisticRegression (multinomial, class_weight=balanced) on StandardScaler'd features, LEAVE-ONE-
SLIDE-OUT over the 3 HD slides. Metrics: macro-F1, accuracy, per-class F1, ECE(max-prob,15 bins), confusion.
RAM-heavy (loads [N,1280..3840] matrices + LR) -> run on a compute node, not a login node.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd

CACHE = "results/cache_pilot"
GT = "results/eval/gt_labels"
HD_SLIDES = ["Slide 6", "Slide 2", "Slide 12"]   # anonymized; see the paper's appendix
SEED = 0


def load_slide(slide, predictor_out, want_pred, fovs, keep=None, cache=CACHE):
    lab = pd.read_parquet(os.path.join(GT, f"{slide}.parquet"))
    if keep is not None:                          # class-reduction filter (e.g. drop NTE)
        lab = lab[lab.L1.isin(keep)].reset_index(drop=True)
    idx = lab.cache_idx.to_numpy()
    feats = {}
    for fov in fovs:
        # NO downsample: fov{N} = frozen Virchow2 m_cell pooled from a bigger 40x-native tile
        # (dynamic pos-embed), same cell order/count across FOVs -> cache_idx is FOV-independent.
        z = np.load(os.path.join(cache, slide, f"fov{fov}", "cells.npz"))
        feats[f"m{fov}"] = np.asarray(z["m_cell"][idx], dtype=np.float32)
    if want_pred:
        p = np.load(os.path.join(predictor_out, f"{slide}.npz"))
        feats["mu"] = np.asarray(p["mu"][idx], dtype=np.float32)
        feats["s2"] = np.asarray(p["s2"][idx], dtype=np.float32)
    return lab.L1.to_numpy(), feats


def build_feature_sets(feats, ablations, want_pred, fovs):
    fs = {f"raw{fov}": feats[f"m{fov}"] for fov in fovs}
    if want_pred:
        base = feats[f"m{fovs[0]}"]
        fs["cellJEPA"] = np.concatenate([base, feats["mu"], feats["s2"]], axis=1)
        # --- 3-arm design (added 2026-08-01) -------------------------------------------------
        # sigma^2 is NOT the same kind of quantity as m_cell/mu: it is a second moment, strictly
        # positive and heavy-tailed, while the others are signed locations in embedding space.
        # StandardScaler (per column, inside each fold) centers and scales but does NOT fix skew.
        # It also risks a SHORTCUT: sigma^2 is context-PREDICTABILITY, so it tracks neighbourhood
        # homogeneity, which correlates with class without being cell identity. So:
        #   cellJEPA_mu     = [m + mu]           <- the clean question: does context help typing?
        #   cellJEPA_logs2  = [m + mu + log s2]  <- sensitivity check, log = what the net predicts
        #   cellJEPA        = [m + mu + s2]      <- kept for continuity with the pre-LoRA number
        # `s2_only` under --ablations is the direct shortcut test: if sigma^2 ALONE types cells
        # above chance, the gain in cellJEPA is partly neighbourhood homogeneity, not identity.
        fs["cellJEPA_mu"] = np.concatenate([base, feats["mu"]], axis=1)
        fs["cellJEPA_logs2"] = np.concatenate(
            [base, feats["mu"], np.log(np.maximum(feats["s2"], 1e-8))], axis=1).astype(np.float32)
        if ablations:
            fs["mu_only"] = feats["mu"]
            fs["s2_only"] = feats["s2"]
            fs["logs2_only"] = np.log(np.maximum(feats["s2"], 1e-8)).astype(np.float32)
            fs["resid"] = base - feats["mu"]
    return fs


def add_l2_arms(fs):
    """Append a per-ROW L2-normalised copy of every feature set (suffix `_l2`).

    Per-VECTOR, not per-dimension. `lopo()` already applies StandardScaler per COLUMN inside each
    fold, so the probe is invariant to any uniform or per-dim rescale -- testing those would measure
    nothing. Projecting each cell onto the unit sphere is the one normalisation StandardScaler
    cannot undo, because it discards each cell's embedding MAGNITUDE. That magnitude is exactly what
    a spherical (von Mises-Fisher) objective would throw away, so this arm prices that design choice
    before we commit to it. Added 2026-07-31.
    """
    out = dict(fs)
    for k, v in fs.items():
        n = np.linalg.norm(v, axis=1, keepdims=True)
        out[f"{k}_l2"] = (v / np.maximum(n, 1e-8)).astype(np.float32)
    return out


def ece(probs, y_idx, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def lopo(Xby, yby, classes, clf_factory, cap=None, seed=SEED):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    rng = np.random.default_rng(seed)
    folds, conf_tot = [], np.zeros((len(classes), len(classes)), dtype=np.int64)
    for held in HD_SLIDES:
        Xtr = np.concatenate([Xby[s] for s in HD_SLIDES if s != held], 0)
        ytr = np.concatenate([yby[s] for s in HD_SLIDES if s != held], 0)
        Xte, yte = Xby[held], yby[held]
        if cap:                                    # kNN is O(N^2) -> subsample; linear stays uncapped
            def _sub(X, y, n):
                if len(y) <= n:
                    return X, y
                sel = np.sort(rng.choice(len(y), n, replace=False))
                return X[sel], y[sel]
            Xtr, ytr = _sub(Xtr, ytr, cap * 2)
            Xte, yte = _sub(Xte, yte, cap)
        sc = StandardScaler().fit(Xtr)
        clf = clf_factory()
        clf.fit(sc.transform(Xtr), ytr)
        pr = clf.predict_proba(sc.transform(Xte))
        cl = list(clf.classes_)
        pred = clf.classes_[pr.argmax(1)]
        y_idx = np.array([cl.index(v) for v in yte])
        folds.append(dict(held=held, n_train=int(len(ytr)), n_test=int(len(yte)),
                          macro_f1=float(f1_score(yte, pred, average="macro", labels=classes)),
                          acc=float(accuracy_score(yte, pred)), ece=ece(pr, y_idx),
                          per_class_f1={c: float(v) for c, v in zip(
                              classes, f1_score(yte, pred, average=None, labels=classes))}))
        conf_tot += confusion_matrix(yte, pred, labels=classes)
    mf1 = np.array([f["macro_f1"] for f in folds])
    acc = np.array([f["acc"] for f in folds])
    return dict(folds=folds, macro_f1_mean=float(mf1.mean()), macro_f1_std=float(mf1.std()),
                acc_mean=float(acc.mean()), ece_mean=float(np.mean([f["ece"] for f in folds])),
                confusion=conf_tot.tolist(), classes=classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor-out", default="results/eval/predictor_out",
                    help="dir with <slide>.npz (mu,s2). If missing, only the raw56 baseline runs.")
    ap.add_argument("--cache", default=CACHE,
                    help="cache root holding <slide>/fov{N}/cells.npz (m_cell). Default results/cache_pilot.")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--max-per-slide", type=int, default=None, help="subsample labelled cells/slide")
    ap.add_argument("--out", default="results/reports")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fovs", default="56",
                    help="comma list of cached FOVs (um) to probe as raw feature sets, e.g. 56,112,168")
    ap.add_argument("--knn", type=int, default=0,
                    help="0=linear LogisticRegression probe; >0=kNN probe with this many neighbors")
    ap.add_argument("--knn-cap", type=int, default=20000,
                    help="per-slide subsample cap when --knn>0 (kNN is O(N^2); the linear probe is uncapped)")
    ap.add_argument("--l2norm-arms", action="store_true",
                    help="ALSO probe a per-row L2-normalised (unit-sphere) copy of each feature set, "
                         "tagged <name>_l2. Default OFF => byte-identical to the pre-2026-07-31 "
                         "behaviour. Tests whether cell embedding MAGNITUDE carries typing signal, "
                         "i.e. what a spherical/vMF objective would discard.")
    ap.add_argument("--keep-classes", default=None,
                    help="comma list of L1 classes to keep (drops all others), e.g. 'Tumor,Immune,Stromal' "
                         "to reduce from the 4-class {Tumor,Immune,Stromal,NTE} to 3 classes. Default=all.")
    args = ap.parse_args()
    fovs = [int(x) for x in str(args.fovs).split(",") if x.strip()]
    keep = set(x.strip() for x in args.keep_classes.split(",") if x.strip()) if args.keep_classes else None

    if args.knn > 0:
        from sklearn.neighbors import KNeighborsClassifier
        clf_factory = lambda: KNeighborsClassifier(n_neighbors=args.knn, metric="cosine",
                                                   weights="distance", n_jobs=-1)
        cap, mode = args.knn_cap, f"knn{args.knn}"
    else:
        from sklearn.linear_model import LogisticRegression
        clf_factory = lambda: LogisticRegression(max_iter=300, C=1.0, multi_class="multinomial",
                                                 class_weight="balanced", n_jobs=-1)
        cap, mode = None, "linear"

    want_pred = all(os.path.exists(os.path.join(args.predictor_out, f"{s}.npz")) for s in HD_SLIDES)
    print(f"mode={mode} fovs={fovs} cap={cap} | cell-JEPA {'AVAILABLE' if want_pred else 'ABSENT -> raw only'}")
    rng = np.random.default_rng(SEED)
    yby, featby = {}, {}
    for s in HD_SLIDES:
        y, feats = load_slide(s, args.predictor_out, want_pred, fovs, keep=keep, cache=args.cache)
        if args.max_per_slide and len(y) > args.max_per_slide:
            sel = np.sort(rng.choice(len(y), args.max_per_slide, replace=False))
            y = y[sel]; feats = {k: v[sel] for k, v in feats.items()}
        yby[s], featby[s] = y, feats
        print(f"  {s}: {len(y):,} labelled cells  {dict(pd.Series(y).value_counts())}")
    classes = sorted(set(np.concatenate([yby[s] for s in HD_SLIDES])))

    results = {}
    for s in HD_SLIDES:
        featby[s] = build_feature_sets(featby[s], args.ablations, want_pred, fovs)
        if args.l2norm_arms:
            featby[s] = add_l2_arms(featby[s])
    for name in featby[HD_SLIDES[0]].keys():
        Xby = {s: featby[s][name] for s in HD_SLIDES}
        r = lopo(Xby, yby, classes, clf_factory, cap=cap)
        results[name] = dict(dim=int(Xby[HD_SLIDES[0]].shape[1]), **r)
        print(f"  {name:10s} dim={results[name]['dim']:5d}  LOPO macro-F1={r['macro_f1_mean']:.3f}"
              f"±{r['macro_f1_std']:.3f}  acc={r['acc_mean']:.3f}  ECE={r['ece_mean']:.3f}")

    ncls = len(classes)
    tag = args.tag or (f"t2_phase5_celltyping_fovsweep_{mode}_{ncls}cls")
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.out, f"{ts}_{tag}")
    os.makedirs(outdir, exist_ok=True)
    summary = dict(generated=ts, cell_jepa_available=want_pred, classes=classes,
                   keep_classes=(sorted(keep) if keep else None),
                   slides=HD_SLIDES, seed=SEED, max_per_slide=args.max_per_slide,
                   fovs=fovs, probe_mode=mode, knn_cap=cap, results=results,
                   note="LOPO=leave-one-slide-out over 3 HD slides; labels=reference L1 (drop Unidentified); "
                        f"keep_classes={sorted(keep) if keep else 'ALL'}; "
                        "raw{N}=frozen Virchow2 m_cell@{N}um (NO downsample: 40x native, dynamic pos-embed); "
                        "probe=linear LogisticRegression or knnK (cosine); kNN subsampled to knn_cap/slide.")
    json.dump(summary, open(os.path.join(outdir, "results.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(outdir, "PROVENANCE.json"), "w"), indent=2)
    _write_md(outdir, summary)
    print(f"\nwrote {outdir}/{{results.json, report.md, PROVENANCE.json}}")


def _write_md(outdir, s):
    L = [f"# P16 T2 Phase 5 — patient-out cell typing (HD)\n",
         f"_generated {s['generated']} · cell-JEPA {'available' if s['cell_jepa_available'] else 'ABSENT (baseline only)'}_\n",
         f"Classes: {s['classes']}  ·  LOPO = leave-one-slide-out over {len(s['slides'])} HD slides\n",
         "| feature set | dim | macro-F1 | acc | ECE |", "|---|---|---|---|---|"]
    for n, r in s["results"].items():
        L.append(f"| {n} | {r['dim']} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} "
                 f"| {r['acc_mean']:.3f} | {r['ece_mean']:.3f} |")
    if "cellJEPA" in s["results"] and "raw56" in s["results"]:
        d = s["results"]["cellJEPA"]["macro_f1_mean"] - s["results"]["raw56"]["macro_f1_mean"]
        L.append(f"\n**cell-JEPA − raw56 macro-F1 = {d:+.3f}**")
    open(os.path.join(outdir, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
