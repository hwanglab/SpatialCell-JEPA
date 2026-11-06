#!/usr/bin/env python3
"""merge_maskctx_split.py — P16 T2 W1: concatenate the per-representation split outputs of
slurm_maskctx_typing_split.sh into the single typing_/slideid_ TSV per arm that
maskctx_paired_report.py reads (glob typing_*.tsv in <ARM>_logistic_pca0/). Asserts completeness.
⚠ slide-ID from split runs is NOT byte-identical to a joint run (sequential rng over arms)."""
import glob, json, os, time, pandas as pd
R = "results/reports/20261104_t2_maskctx_typing"
REPS = ["m", "mu", "r", "smooth", "m+mu", "m+smooth", "m+r"]
for arm in ["R0_none", "R1_target", "R2_control", "Efixed_target"]:
    d = f"{R}/{arm}_logistic_pca0"; tag = f"t2_maskctx_{arm}_fov56_logistic_pca0"
    ty, sl, src = [], [], []
    for rep in REPS:
        sd = f"{d}/split/{rep.replace('+', 'p')}"
        f = glob.glob(f"{sd}/typing_*.tsv"); g = glob.glob(f"{sd}/slideid_*.tsv")
        assert len(f) == 1 and len(g) == 1, f"{arm}/{rep}: missing split output in {sd}"
        t = pd.read_csv(f[0], sep="\t"); assert set(t.arm) == {rep}, (arm, rep, set(t.arm))
        assert len(t) == 12 * 3, f"{arm}/{rep}: {len(t)} rows, expected 36 (12 folds x 3 C)"
        ty.append(t); sl.append(pd.read_csv(g[0], sep="\t")); src += [f[0], g[0]]
    ty = pd.concat(ty, ignore_index=True); sl = pd.concat(sl, ignore_index=True)
    m = ty[(ty.arm == "m") & (ty.C == 0.01)].macro_f1.mean()
    assert abs(m - 0.6159) < 5e-4, f"{arm}: sanity m = {m:.4f} != 0.6159"
    ty.to_csv(f"{d}/typing_{tag}.tsv", sep="\t", index=False)
    sl.to_csv(f"{d}/slideid_{tag}.tsv", sep="\t", index=False)
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": os.path.abspath(__file__),
               "inputs": {p: dict(mtime=os.path.getmtime(p), size=os.path.getsize(p)) for p in src},
               "note": "merged from one-representation-per-task split runs; typing identical to joint "
                       "run; slide-ID NOT byte-identical (sequential rng over arms)"},
              open(f"{d}/PROVENANCE_merge_{tag}.json", "w"), indent=1)
    print(f"{arm}: merged {len(ty)} typing rows, m(C=0.01) = {m:.4f}")
