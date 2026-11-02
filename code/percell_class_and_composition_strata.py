#!/usr/bin/env python3
"""percell_class_and_composition_strata.py — does `mu` differ from `m` BY CLASS, and BY SLIDE TYPE?

Answers workshop-paper headline (c-2) [user 2026-08-24]:
  "we need to find what differences between two represations. I think we did classification
   perfomrance anlayssi for each cell type. they are the same between two representations?
   can we divide the slides into tumor dominant or stromal any further division so that can
   we find any pattern?"

★ NO NEW COMPUTE AND NO PER-CELL DUMP NEEDED. Both halves are already on disk:
  - per-fold, per-arm, PER-CLASS F1 -> `typing_corpus31_fov56.tsv` (cols f1_Immune/f1_Stromal/
    f1_Tumor, one row per arm x estimator x C x held-out slide)
  - per-slide class composition -> `results/eval/gt_labels_hd12/<slide>.parquet` (col `L1`)
  The `held` column is the held-out slide id, so the two join on slide.
  (This is why c-2 does NOT hit the ceiling-attribution option-A blocker: that blocker is about
   per-CELL predictions; per-CLASS F1 is an aggregate the readout already writes.)

PARSES ONLY -- every number is read from a named artifact, none recomputed from features and none
retyped. Read-only on all inputs.

DESIGN (stated before any number is inspected, per the project's ARMS rule):
  estimator : multinomial logistic, C = 0.01 FIXED (not best_C, which is selected on held-out
              folds and optimistically biased), class_weight balanced -- whatever
              p2_typing_and_slideid.py used to write the TSV. NOT refitted here.
  input X   : `m`  = the cell's own frozen H-optimus-0 nucleus-hull-pooled token, 1536-d
              `mu` = the predictor's estimate from the 36 nearest neighbours; the target's own
                     token is excluded from its context, so `mu` never observes the cell
  output y  : L1 3-class {Immune, Stromal, Tumor}, RNA-derived, evaluation only
  unit / n  : one cell; 12,000 cells per held-out slide
  split     : leave-one-patient-out, 12 folds, fit on 11 slides, score on the 12th
  pairing   : `m` and `mu` share cells, folds, seed and harness => PAIRED per fold
  stratifier: per-slide tumour fraction from the GT labels. A SLIDE property computed from the
              label file, independent of either arm and of any prediction => cannot leak.

⚠ The stratification is a SLIDE-LEVEL split with 12 slides, so each stratum holds ~6 folds.
  Wilcoxon on n=6 bottoms out at p = 0.0312 two-sided. Report direction + effect size; a stratum
  difference here is a HYPOTHESIS, not a finding. MDE80 is printed so the reader can see it.
⚠ Composition is computed over ALL labelled cells in the slide, not the 12,000 the probe sampled.
  Stated, not silently assumed equivalent.
"""
import argparse, glob, json, os, hashlib, datetime
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CLASSES = ["Immune", "Stromal", "Tumor"]


def md5(p, cap=64 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(1 << 20)):
            h.update(b)
            if f.tell() > cap:
                break
    return h.hexdigest()


def mde80(d):
    """Smallest true effect this paired design would detect 80% of the time, two-sided a=0.05."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return np.nan
    return float(2.802 * d.std(ddof=1) / np.sqrt(n))


def paired(delta):
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return dict(mean=np.nan, folds_pos=0, n=len(d), p=np.nan, mde80=np.nan)
    try:
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = np.nan
    return dict(mean=float(d.mean()), folds_pos=int((d > 0).sum()), n=int(len(d)),
                p=p, mde80=mde80(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typing-tsv", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--estimator", default="logistic")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- per-slide composition, from the GT labels only -----------------------------------
    comp = []
    gt_files = sorted(glob.glob(os.path.join(a.gt_dir, "*.parquet")))
    assert gt_files, f"no parquet in {a.gt_dir}"
    for f in gt_files:
        d = pd.read_parquet(f, columns=["L1"])
        n = len(d)
        vc = d["L1"].value_counts()
        row = {"held": os.path.basename(f).replace(".parquet", ""), "n_labelled": n}
        for c in CLASSES:
            row[f"frac_{c}"] = float(vc.get(c, 0)) / n
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp["tumor_stroma_ratio"] = comp["frac_Tumor"] / comp["frac_Stromal"]
    med = comp["frac_Tumor"].median()
    comp["stratum"] = np.where(comp["frac_Tumor"] >= med, "tumor_dominant", "stroma_dominant")
    comp = comp.sort_values("frac_Tumor", ascending=False)

    # ---- typing: keep the fixed-C readout cell, arms m and mu -----------------------------
    t = pd.read_csv(a.typing_tsv, sep="\t")
    t = t[(t["estimator"] == a.estimator) & (np.isclose(t["C"], a.C))]
    t = t[t["arm"].isin(["m", "mu"])]
    metrics = ["macro_f1", "acc"] + [f"f1_{c}" for c in CLASSES]
    w = t.pivot_table(index="held", columns="arm", values=metrics)
    # flatten MultiIndex ("macro_f1","mu") -> "mu_macro_f1" so the join stays single-level
    w.columns = [f"{arm}_{mt}" for mt, arm in w.columns]
    w = w.reset_index()
    assert len(w) == 12, f"expected 12 held-out slides, got {len(w)}"
    for mt in metrics:
        for arm in ("m", "mu"):
            assert f"{arm}_{mt}" in w.columns, f"missing {arm}_{mt} in the typing TSV"

    j = w.merge(comp[["held", "frac_Tumor", "frac_Stromal", "frac_Immune",
                      "tumor_stroma_ratio", "stratum"]], on="held", how="inner")
    assert len(j) == 12, f"join lost slides: {len(j)}/12"

    # ---- deltas -----------------------------------------------------------------------------
    for mt in metrics:
        j[f"d_{mt}"] = j[f"mu_{mt}"] - j[f"m_{mt}"]

    rows = []
    for mt in metrics:
        rows.append(dict(scope="ALL", n_slides=len(j), metric=mt,
                         m=float(j[f"m_{mt}"].mean()), mu=float(j[f"mu_{mt}"].mean()),
                         **paired(j[f"d_{mt}"])))
    for st, g in j.groupby("stratum"):
        for mt in metrics:
            rows.append(dict(scope=st, n_slides=len(g), metric=mt,
                             m=float(g[f"m_{mt}"].mean()), mu=float(g[f"mu_{mt}"].mean()),
                             **paired(g[f"d_{mt}"])))
    res = pd.DataFrame(rows)

    # ---- does the per-slide delta track composition? (continuous, no binning) --------------
    corr = {}
    for mt in metrics:
        v = j[[f"d_{mt}", "frac_Tumor"]].dropna()
        if len(v) >= 3:
            corr[mt] = {
                "spearman_d_vs_fracTumor": float(v.corr(method="spearman").iloc[0, 1]),
                "pearson_d_vs_fracTumor": float(v.corr(method="pearson").iloc[0, 1]),
                "n_slides": int(len(v))}

    comp.to_csv(os.path.join(a.out_dir, "slide_composition.tsv"), sep="\t", index=False)
    res.to_csv(os.path.join(a.out_dir, "per_class_and_strata.tsv"), sep="\t", index=False)
    j.to_csv(os.path.join(a.out_dir, "per_slide_paired.tsv"), sep="\t", index=False)

    prov = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": os.path.abspath(__file__),
        "maturity": "[FINAL n=12/12 slides]",
        "inputs": {
            "typing_tsv": {"path": os.path.abspath(a.typing_tsv), "md5": md5(a.typing_tsv),
                           "size": os.path.getsize(a.typing_tsv)},
            "gt_dir": {"path": os.path.abspath(a.gt_dir), "n_parquet": len(gt_files),
                       "md5_each": {os.path.basename(f): md5(f) for f in gt_files}},
        },
        "readout_cell": {"estimator": a.estimator, "C": a.C,
                         "note": "C FIXED, not best_C; arms m and mu only"},
        "stratifier": {"rule": "per-slide GT tumour fraction, split at the 12-slide median",
                       "median_frac_Tumor": float(med),
                       "computed_over": "ALL labelled cells in the slide, not the 12,000 sampled"},
        "caveats": [
            "6 folds per stratum => Wilcoxon floor p=0.0312 two-sided; direction + effect size only.",
            "A stratum difference here is a hypothesis, not a finding; MDE80 reported per row.",
            "Per-CLASS F1 is an aggregate the readout already writes; this does NOT need the "
            "per-cell prediction dump that blocks ceiling-attribution option A.",
        ],
        "spearman_delta_vs_composition": corr,
    }
    with open(os.path.join(a.out_dir, "PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2)

    pd.set_option("display.width", 200)
    print("### per-slide composition\n", comp.to_string(index=False))
    print("\n### paired m vs mu, ALL and by stratum\n", res.to_string(index=False))
    print("\n### delta vs tumour fraction\n", json.dumps(corr, indent=2))
    print("\nwrote", a.out_dir)


if __name__ == "__main__":
    main()
